# cassava_training_single_fold.py
"""
Cassava Leaf Disease Classification – **Single‑Fold (fold‑0) Training**
=====================================================================
• 所有保留樣本 (label ≠ 3) 均標記為 `fold = 0`，一次性訓練 + 驗證。
• 不做 5‑fold 迴圈；最佳權重存為 `outputs/best_fold0.pth`。
• 其餘邏輯（過濾 label==3、MixUp、GeM、log 檔）維持不變。
"""

from __future__ import annotations
import os, math, random, time, logging, warnings
from pathlib import Path
from typing import List
from sklearn.model_selection import StratifiedShuffleSplit
import cv2, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from albumentations import (Compose, RandomResizedCrop, HorizontalFlip, VerticalFlip,
                            ShiftScaleRotate, CoarseDropout, Normalize, Resize, Transpose)
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
import timm


warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CFG:
    seed = 42
    num_epochs = 30
    batch_size = 64
    num_workers = 4

    backbone = "efficientnet_b5.sw_in12k_ft_in1k"
    num_classes = 4
    drop_path_rate = 0.1
    lr = 1e-4; weight_decay = 1e-3; max_grad_norm = 1000
    scheduler = "cosine"; t_max = 30; min_lr = 1e-6
    mixup_prob = 0.8; mixup_alpha = 0.4; cutmix_alpha = 1.0; label_smoothing = 0.1

    img_size = 448
    print_freq = 100
    output_dir = Path("./outputs")
    train_path = Path("cassava-leaf-disease-classification/train_images")

CFG.output_dir.mkdir(exist_ok=True)


# logging -----------------------------------------------------------------
log_path = CFG.output_dir / "train.log"
logging.basicConfig(filename=log_path, level=logging.INFO,
                    format="%(asctime)s %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logging.getLogger("").addHandler(logging.StreamHandler())


# seed ---------------------------------------------------------------------
random.seed(CFG.seed); np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed); torch.cuda.manual_seed_all(CFG.seed)


# utils --------------------------------------------------------------------
class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val=self.avg=self.sum=self.count=0.
    def update(self,val,n=1): self.val=val; self.sum+=val*n; self.count+=n; self.avg=self.sum/self.count

def get_score(y_true,y_pred): return accuracy_score(y_true,y_pred)

def time_since(since,progress): now=time.time(); elapsed=now-since; total=elapsed/progress; remain=total-elapsed; m=int(elapsed//60); return f"{m}m {int(elapsed-m*60)}s (remain {int(remain//60)}m {int(remain%60)}s)"


# dataset ------------------------------------------------------------------
class CassavaDataset(Dataset):
    def __init__(self,df,stage):
        self.paths=df.image_id.values; self.labels=df.label.values
        self.transform=get_transforms(stage)
    def __len__(self): return len(self.paths)
    def __getitem__(self,idx):
        img=cv2.cvtColor(cv2.imread(str(CFG.train_path/self.paths[idx])),cv2.COLOR_BGR2RGB)
        if self.transform: img=self.transform(image=img)["image"]
        label=int(self.labels[idx]); label=3 if label==4 else label
        return img, torch.tensor(label).long()

def get_transforms(stage):
    if stage=="train":
        return Compose([RandomResizedCrop((CFG.img_size,CFG.img_size)), Transpose(p=.5), HorizontalFlip(p=.5), VerticalFlip(p=.5),
                        ShiftScaleRotate(p=.5), CoarseDropout(max_holes=8,max_height=CFG.img_size//8,max_width=CFG.img_size//8,p=.5),
                        Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]), ToTensorV2()])
    return Compose([Resize(CFG.img_size,CFG.img_size), Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]), ToTensorV2()])


# model --------------------------------------------------------------------
class GeM(nn.Module):
    def __init__(self,p=3.,eps=1e-6): super().__init__(); self.p=nn.Parameter(torch.ones(1)*p); self.eps=eps
    def forward(self,x): return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p),1).pow(1./self.p)
class NetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=timm.create_model(CFG.backbone,pretrained=True,features_only=True,out_indices=[-1],drop_path_rate=CFG.drop_path_rate)
        in_ch=self.backbone.feature_info.channels()[-1]
        self.pool=GeM(); self.head=nn.Sequential(nn.Linear(in_ch,256),nn.SiLU(),nn.Dropout(.3),nn.Linear(256,CFG.num_classes))
    def forward(self,x): x=self.backbone(x)[0]; x=self.pool(x).flatten(1); return self.head(x)


# mixup / loss -------------------------------------------------------------
mixup_fn=Mixup(mixup_alpha=CFG.mixup_alpha,cutmix_alpha=CFG.cutmix_alpha,prob=CFG.mixup_prob,switch_prob=.5,mode="batch",label_smoothing=CFG.label_smoothing,num_classes=CFG.num_classes)
criterion_soft=SoftTargetCrossEntropy(); criterion_hard=nn.CrossEntropyLoss()


# train / val --------------------------------------------------------------
def train_epoch(loader, model, opt, sched, epoch):
    model.train()
    meter = AverageMeter()
    start = time.time()
    for step, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        imgs, targets = mixup_fn(imgs, labels)
        loss = criterion_soft(model(imgs), targets)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        opt.step()
        meter.update(loss.item(), imgs.size(0))
    sched.step()
    logging.info(f"Epoch {epoch+1:02d} – Train Loss {meter.avg:.4f} – Time {time_since(start, 1.0)}")
    return meter.avg


def valid_epoch(loader,model):
    model.eval(); meter=AverageMeter(); preds=[]; labels_all=[]
    with torch.no_grad():
        for imgs,labels in loader:
            imgs,labels=imgs.to(DEVICE),labels.to(DEVICE); out=model(imgs); loss=criterion_hard(out,labels)
            meter.update(loss.item(),imgs.size(0)); preds.append(torch.softmax(out,1).cpu().numpy()); labels_all.append(labels.cpu().numpy())
    preds=np.concatenate(preds); labels=np.concatenate(labels_all); return meter.avg,get_score(labels,preds.argmax(1)),preds


# main ---------------------------------------------------------------------
def main():
    df=pd.read_csv("cassava-leaf-disease-classification/train.csv")
    df=df[df.label!=3].reset_index(drop=True)
    df["fold"]=0
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=CFG.seed)
    train_idx, val_idx = next(sss.split(df, df['label']))
    train_ds=CassavaDataset(df.loc[train_idx],"train")
    val_ds  =CassavaDataset(df.loc[val_idx],"valid")

    train_ld=DataLoader(train_ds,batch_size=CFG.batch_size,shuffle=True,num_workers=CFG.num_workers,pin_memory=True,drop_last=True)
    val_ld  =DataLoader(val_ds,batch_size=CFG.batch_size,shuffle=False,num_workers=CFG.num_workers,pin_memory=True)

    model=nn.DataParallel(NetClassifier()).to(DEVICE)
    opt=AdamW(model.parameters(),lr=CFG.lr,weight_decay=CFG.weight_decay)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=CFG.t_max,eta_min=CFG.min_lr)

    best_acc=0.; best_preds=None
    for ep in range(CFG.num_epochs):
        tl=train_epoch(train_ld,model,opt,sched,ep)
        vl,acc,preds=valid_epoch(val_ld,model)
        logging.info(f"Ep{ep+1:02d} TL {tl:.4f} VL {vl:.4f} ACC {acc:.4f}")
        if acc>best_acc:
            best_acc, best_preds = acc, preds
            torch.save(model.state_dict(), CFG.output_dir/"best_fold0.pth")

    # Save only validation set predictions as oof
    oof_val_df = df.loc[val_idx].copy()
    oof_val_df.reset_index(drop=True, inplace=True)
    oof_val_df[[str(i) for i in range(CFG.num_classes)]] = best_preds
    oof_val_df["preds"] = best_preds.argmax(1)
    oof_val_df.to_csv(CFG.output_dir / "oof_val_df.csv", index=False)

    logging.info(f"Final ACC on validation set: {best_acc:.4f}")




if __name__=="__main__":
    main()
