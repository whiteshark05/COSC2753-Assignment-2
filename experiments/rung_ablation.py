"""Task 4 ablation harness — one change per run, everything else held fixed.

Rebuilds the EXACT gallery/query split from data/processed/ artifacts (written by
COSC2753_A2_Task4.ipynb), so every rung is comparable to the notebook's numbers.

  --stem default   reproduces the notebook's model (rung 0, mAP@10 = 0.7960)
  --stem cifar     rung 1: 3x3 stride-1 stem, no maxpool
  --pool gem       rung 2: generalised-mean pooling (learnable p) instead of GAP
  --head colour    rung 3: 256-d = 192 semantic + 64 colour, second triplet on baseColour
  --eval-only PATH scores an existing checkpoint (harness validation)

Defaults reproduce rung 0 exactly; pass one flag at a time.
"""
import argparse, json, time, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms as T
from torchvision.models import resnet18
from PIL import Image
from tqdm.auto import tqdm

REPO = Path(__file__).resolve().parent.parent
# Task 4 scores against its OWN split artifacts. data/processed is the shared,
# git-tracked preprocessing for Tasks 1-3; a teammate regenerating it silently
# changed this harness's gallery/query sets once already (28,958/9,648 ->
# 28,371/9,455), which would invalidate every cross-rung comparison.
PROC = REPO / "data" / "processed_task4"
IMAGES = REPO / "data" / "raw" / "FashionDataset" / "train" / "images_train"
RANDOM_STATE = 42
K_VALUES, K_PRIMARY = [1, 5, 10, 20], 10
EMB_DIM, MARGIN = 128, 0.2
P_CLASSES, K_PER_CLASS, BATCHES_PER_EPOCH = 16, 4, 450
# Standard budget for the ablation ladder: 50 epochs. Matches the group's house
# setting (Task 1's EPOCHS=50, Task 3's real config) and makes rung 1 -- stopped at
# exactly 50 -- a matched-budget result rather than a truncated one. Rung 0 must be
# re-baselined at 50: its 80-epoch score of 0.7961 is worth ~0.019 more than its
# best-within-50, so scoring 50-epoch rungs against it would penalise every rung.
LR, EPOCHS, PATIENCE = 3e-4, 50, 8
VAL_PROBE_N = 1500
AUX_COLS = ['subCategory', 'gender', 'season', 'usage_grouped']  # rung 4


def device():
    if torch.cuda.is_available(): return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')
DEVICE = device()

# ── data ────────────────────────────────────────────────────────────────────
cfg = json.load(open(PROC / "pipeline_config.json"))
IMG_W, IMG_H = cfg["image"]["width"], cfg["image"]["height"]
MEAN, STD = cfg["image"]["normalization_mean"], cfg["image"]["normalization_std"]

def to_rgb(im): return im.convert("RGB")     # named, not a lambda (picklable)

eval_transform = T.Compose([T.Lambda(to_rgb), T.Resize((IMG_H, IMG_W)),
                            T.ToTensor(), T.Normalize(MEAN, STD)])
train_transform = T.Compose([T.Lambda(to_rgb), T.Resize((IMG_H, IMG_W)),
                             T.RandomHorizontalFlip(p=0.5), T.RandomRotation(degrees=10),
                             T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                             T.ToTensor(), T.Normalize(MEAN, STD)])

def load_frames():
    g = pd.read_csv(PROC / "holdout_metadata" / "articleType_train.csv")
    q = pd.read_csv(PROC / "holdout_metadata" / "articleType_val.csv")
    for d in (g, q):
        d['id'] = d['id'].astype(str)
    assert set(g.id).isdisjoint(set(q.id)) and set(g.dup_group).isdisjoint(set(q.dup_group))
    return g.reset_index(drop=True), q.reset_index(drop=True)

def build_cache(ids):
    ids = list(dict.fromkeys(ids))
    cache = np.zeros((len(ids), IMG_H, IMG_W, 3), dtype=np.uint8)
    index = {}
    for i, img_id in enumerate(tqdm(ids, desc="caching", leave=False)):
        with Image.open(IMAGES / f"{img_id}.jpg") as im:
            cache[i] = np.asarray(im.convert("RGB").resize((IMG_W, IMG_H)), dtype=np.uint8)
        index[str(img_id)] = i
    return cache, index

class CachedDS(Dataset):
    def __init__(self, df, cache, index, transform, label_col=None, aux_cols=()):
        self.df, self.cache, self.index = df.reset_index(drop=True), cache, index
        self.transform, self.label_col, self.aux_cols = transform, label_col, list(aux_cols)
        self.rows = np.array([index[str(i)] for i in self.df['id']])
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        img = self.transform(Image.fromarray(self.cache[self.rows[i]]))
        row = self.df.iloc[i]
        lab = 0 if self.label_col is None else int(row[self.label_col])
        aux = torch.tensor([int(row[c]) for c in self.aux_cols], dtype=torch.long)
        return img, torch.tensor(lab, dtype=torch.long), aux

class PKSampler(Sampler):
    def __init__(self, labels, P=P_CLASSES, K=K_PER_CLASS, batches=BATCHES_PER_EPOCH, seed=RANDOM_STATE):
        self.labels = np.asarray(labels); self.P, self.K, self.batches = P, K, batches
        self.rng = np.random.default_rng(seed)
        self.c2i = {c: np.flatnonzero(self.labels == c) for c in np.unique(self.labels)}
        self.usable = [c for c, idx in self.c2i.items() if len(idx) >= 2]
    def __iter__(self):
        for _ in range(self.batches):
            chosen = self.rng.choice(self.usable, size=min(self.P, len(self.usable)), replace=False)
            batch = []
            for c in chosen:
                pool = self.c2i[c]
                batch.extend(self.rng.choice(pool, size=self.K, replace=len(pool) < self.K))
            yield [int(i) for i in batch]
    def __len__(self): return self.batches

# ── model ───────────────────────────────────────────────────────────────────
class GeM(nn.Module):
    """Generalised-mean pooling: (mean x^p)^(1/p), p learnable and shared across
    channels. p=1 is average pooling, p->inf is max pooling, so GAP is inside the
    family -- the layer can only help if the optimiser moves p off 1.
    p must be excluded from weight decay (decay pulls it toward the degenerate 0)."""
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__(); self.p = nn.Parameter(torch.tensor(float(p))); self.eps = eps
    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), 1).pow(1.0 / self.p)


class ResNetStyleCNN(nn.Module):
    """weights=None throughout -- fully self-trained, per the assignment.
    stem='cifar' replaces the 7x7/s2 + maxpool front end with a 3x3/s1 conv,
    keeping 80x60 through layer1 instead of collapsing it to 20x15."""
    def __init__(self, out_dim=EMB_DIM, stem='default', pool='gap'):
        super().__init__()
        b = resnet18(weights=None)
        if stem == 'cifar':
            b.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            b.maxpool = nn.Identity()
        elif stem != 'default':
            raise ValueError(stem)
        if pool == 'gem':
            b.avgpool = GeM()
        elif pool != 'gap':
            raise ValueError(pool)
        b.fc = nn.Identity()
        self.backbone, self.out_dim = b, out_dim
        self.proj = nn.Linear(512, out_dim)
    def forward(self, x): return self.proj(self.backbone(x))

class EmbeddingNet(nn.Module):
    def __init__(self, enc, aux_dims=(), arcface_classes=None):
        super().__init__(); self.encoder, self.out_dim = enc, enc.out_dim
        # rung 4: linear CE heads on four labelled columns the notebook never used.
        # They are supervision only -- retrieval still uses the embedding itself.
        self.aux_heads = nn.ModuleList([nn.Linear(enc.out_dim, d) for d in aux_dims])
        self.arc = ArcFace(enc.out_dim, arcface_classes) if arcface_classes else None
    def forward(self, x): return F.normalize(self.encoder(x), p=2, dim=1)

class ArcFace(nn.Module):
    """Rung 5. Additive angular margin on the target class (Deng et al., 2019). The
    classification and metric models tied to within 0.0002, which suggests they encode
    complementary things; ArcFace is the natural hybrid because it is a classification
    loss that operates on the same normalised sphere the triplet does, so the two can
    share one embedding rather than fighting over it.
    margin 0.2 rather than the usual 0.5: this backbone trains from scratch, and a large
    angular margin early in training is what makes ArcFace diverge."""
    def __init__(self, dim, n_classes, scale=30.0, margin=0.2):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, dim) * 0.01)
        self.scale, self.margin = scale, margin
    def forward(self, emb, labels):
        cos = (emb @ F.normalize(self.W, p=2, dim=1).t()).clamp(-1 + 1e-7, 1 - 1e-7)
        onehot = torch.zeros_like(cos).scatter_(1, labels[:, None], 1.0)
        return torch.cos(torch.acos(cos) + self.margin * onehot) * self.scale


class ColourEmbeddingNet(nn.Module):
    """Rung 3. 256-d = 192 semantic + 64 colour. Each block is L2-normalised on its
    own and the concatenation scaled by 1/sqrt(2), so the full vector stays unit-norm
    and cosine similarity is exactly the MEAN of the semantic and colour cosines --
    colour gets a fixed, interpretable half of the retrieval score instead of
    whatever share gradient descent happens to leave it."""
    SEM, COL = 192, 64
    def __init__(self, enc, n_colours=None, ce_scale=16.0, arcface_classes=None):
        super().__init__(); self.encoder, self.out_dim = enc, enc.out_dim
        assert enc.out_dim == self.SEM + self.COL, enc.out_dim
        # rung 3b: cosine classifier over the colour block. A batch-hard triplet on
        # baseColour collapses (rung 3: loss pinned at the margin, every colour vector
        # identical) because hardest-negative mining keeps drawing Blue/Navy Blue and
        # Grey/Silver pairs that are not actually different colours. Cross-entropy has
        # no such fixed point -- a constant embedding gives uniform logits and MAXIMAL
        # loss, so the gradient always pushes the classes apart.
        self.ce_scale, self.ce_W = ce_scale, None
        if n_colours:
            self.ce_W = nn.Parameter(torch.randn(n_colours, self.COL) * 0.01)
        # rung 6: the angular margin applies to the SEMANTIC block only -- colour is
        # supervised by its own CE, and putting both on the same block would just
        # recreate the weight contest that collapsed rungs 3 and 3b.
        self.arc = ArcFace(self.SEM, arcface_classes) if arcface_classes else None
    def colour_logits(self, col_block):
        return self.ce_scale * (col_block @ F.normalize(self.ce_W, p=2, dim=1).t())
    def forward(self, x):
        z = self.encoder(x)
        s = F.normalize(z[:, :self.SEM], p=2, dim=1)
        c = F.normalize(z[:, self.SEM:], p=2, dim=1)
        return torch.cat([s, c], dim=1) / (2 ** 0.5)
    @staticmethod
    def blocks(emb):
        """Recover the unit-norm semantic and colour blocks from a full embedding."""
        r = 2 ** 0.5
        return emb[:, :ColourEmbeddingNet.SEM] * r, emb[:, ColourEmbeddingNet.SEM:] * r


def batch_hard_triplet_loss(emb, labels, margin=MARGIN):
    dist = torch.cdist(emb, emb, p=2)
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=emb.device)
    pos, neg = same & ~eye, ~same
    d_pos = dist.masked_fill(~pos, -1.0).max(dim=1).values
    d_neg = dist.masked_fill(~neg, float('inf')).min(dim=1).values
    valid = pos.any(dim=1) & neg.any(dim=1)
    losses = F.relu(d_pos - d_neg + margin)[valid]
    return losses.mean(), float((losses > 0).float().mean() if len(losses) else 0.0)

# ── metrics (identical to the notebook) ─────────────────────────────────────
def l2n(x): return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

@torch.no_grad()
def embed(model, df, cache, index, bs=256):
    model.eval()
    dl = DataLoader(CachedDS(df, cache, index, eval_transform), batch_size=bs, shuffle=False, num_workers=0)
    return l2n(torch.cat([model(b[0].to(DEVICE)).float().cpu() for b in dl]).numpy())

def search_topk(q, g, k=max(K_VALUES), chunk=1024):
    out = np.empty((len(q), k), dtype=np.int64); gt = g.T.astype(np.float32)
    for s in range(0, len(q), chunk):
        e = min(s + chunk, len(q))
        sims = q[s:e].astype(np.float32) @ gt
        part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sims, part, axis=1), axis=1)
        out[s:e] = np.take_along_axis(part, order, axis=1)
    return out

def metrics(topk, q_at, q_bc, g_at, g_bc, at_counts, pair_counts, nbc, ks=K_VALUES, strict=False):
    rel = (g_at[topk] == q_at[:, None])
    if strict: rel &= (g_bc[topk] == q_bc[:, None])
    if strict:
        keys = q_at.astype(np.int64) * (nbc + 1) + q_bc
        n_rel = np.array([pair_counts.get(int(k), 0) for k in keys])
    else:
        n_rel = at_counts[q_at]
    out = {}
    for k in ks:
        rk = rel[:, :k]
        out[f'P@{k}'] = rk.mean(axis=1)
        csum = np.cumsum(rk, axis=1); ranks = np.arange(1, k + 1)[None, :]
        out[f'AP@{k}'] = ((csum / ranks) * rk).sum(axis=1) / np.minimum(k, np.maximum(n_rel, 1))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stem', default='default', choices=['default', 'cifar'])
    ap.add_argument('--pool', default='gap', choices=['gap', 'gem'])
    ap.add_argument('--head', default='plain', choices=['plain', 'colour'])
    ap.add_argument('--aux', action='store_true',
                    help='rung 4: CE heads on subCategory/gender/season/usage')
    ap.add_argument('--aux-weight', type=float, default=0.03)
    ap.add_argument('--arcface', action='store_true', help='rung 5: ArcFace + triplet')
    ap.add_argument('--arc-weight', type=float, default=0.10)
    ap.add_argument('--colour-loss', default='triplet', choices=['triplet', 'ce'],
                    help='supervision on the colour block (head=colour only)')
    ap.add_argument('--colour-weight', type=float, default=1.0,
                    help='weight on the baseColour loss (head=colour only)')
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--eval-only', default=None)
    args = ap.parse_args()
    tag = args.tag or f"stem-{args.stem}-pool-{args.pool}-head-{args.head}"
    assert not (args.aux and args.arcface), "one change per run"
    assert not (args.head == 'colour' and args.aux), "one change per run"
    # colour + arcface IS rung 6 -- the deliberate combination of the two winners.

    gal, qry = load_frames()
    print(f"gallery {len(gal):,}  queries {len(qry):,}  device {DEVICE}")
    cache, index = build_cache(list(gal['id']) + list(qry['id']))
    print(f"cache {cache.shape} ({cache.nbytes/1e6:.0f} MB)")

    at_codes = {c: i for i, c in enumerate(pd.concat([gal.articleType_grouped, qry.articleType_grouped]).astype(str).unique())}
    bc_codes = {c: i for i, c in enumerate(pd.concat([gal.baseColour, qry.baseColour]).astype(str).unique())}
    g_at = gal.articleType_grouped.astype(str).map(at_codes).to_numpy()
    g_bc = gal.baseColour.astype(str).map(bc_codes).to_numpy()
    q_at = qry.articleType_grouped.astype(str).map(at_codes).to_numpy()
    q_bc = qry.baseColour.astype(str).map(bc_codes).to_numpy()
    gal['baseColour_enc'] = g_bc          # rung 3's second supervision signal
    aux_dims = []
    for c in AUX_COLS:                     # rung 4's four unused labelled columns
        codes = {v: i for i, v in enumerate(gal[c].astype(str).unique())}
        gal[c + '_enc'] = gal[c].astype(str).map(codes)
        aux_dims.append(len(codes))
    at_counts = np.bincount(g_at, minlength=len(at_codes))
    pk = g_at.astype(np.int64) * (len(bc_codes) + 1) + g_bc
    pair_counts = {int(a): int(b) for a, b in zip(*np.unique(pk, return_counts=True))}
    NBC = len(bc_codes)

    def full_eval(model):
        ge, qe = embed(model, gal, cache, index), embed(model, qry, cache, index)
        tk = search_topk(qe, ge)
        s = metrics(tk, q_at, q_bc, g_at, g_bc, at_counts, pair_counts, NBC, strict=False)
        st = metrics(tk, q_at, q_bc, g_at, g_bc, at_counts, pair_counts, NBC, strict=True)
        rng = np.random.default_rng(RANDOM_STATE); v = s[f'AP@{K_PRIMARY}']; n = len(v)
        boot = np.array([v[rng.integers(0, n, n)].mean() for _ in range(1000)])
        return dict(
            **{f'P@{k}': float(s[f'P@{k}'].mean()) for k in K_VALUES},
            mAP10=float(s['AP@10'].mean()),
            ci_low=float(np.quantile(boot, .025)), ci_high=float(np.quantile(boot, .975)),
            mAP10_strict=float(st['AP@10'].mean()), P10_strict=float(st['P@10'].mean()))

    torch.manual_seed(RANDOM_STATE)
    if args.head == 'colour':
        dim = ColourEmbeddingNet.SEM + ColourEmbeddingNet.COL
        model = ColourEmbeddingNet(ResNetStyleCNN(dim, stem=args.stem, pool=args.pool),
                                   n_colours=len(bc_codes) if args.colour_loss == 'ce' else None,
                                   arcface_classes=len(at_codes) if args.arcface else None)
    else:
        model = EmbeddingNet(ResNetStyleCNN(EMB_DIM, stem=args.stem, pool=args.pool),
                             aux_dims=aux_dims if args.aux else (),
                             arcface_classes=len(at_codes) if args.arcface else None)
    model = model.to(DEVICE)
    nparam = sum(p.numel() for p in model.parameters())

    if args.eval_only:
        model.load_state_dict(torch.load(args.eval_only, map_location=DEVICE))
        r = full_eval(model)
        print("\n=== EVAL ONLY (harness validation) ===")
        print(json.dumps(r, indent=2))
        return

    probe = qry.sample(min(VAL_PROBE_N, len(qry)), random_state=RANDOM_STATE)
    p_at = qry.articleType_grouped.astype(str).map(at_codes).to_numpy()[probe.index.to_numpy()]
    p_bc = qry.baseColour.astype(str).map(bc_codes).to_numpy()[probe.index.to_numpy()]
    probe = probe.reset_index(drop=True)

    def probe_map10(m):
        ge, qe = embed(m, gal, cache, index), embed(m, probe, cache, index)
        tk = search_topk(qe, ge, k=K_PRIMARY)
        return float(metrics(tk, p_at, p_bc, g_at, g_bc, at_counts, pair_counts, NBC, ks=[K_PRIMARY])['AP@10'].mean())

    aux_cols = ['baseColour_enc'] if args.head == 'colour' else \
               [c + '_enc' for c in AUX_COLS] if args.aux else []
    loader = DataLoader(CachedDS(gal, cache, index, train_transform,
                                 'articleType_grouped_enc', aux_cols),
                        batch_sampler=PKSampler(gal.articleType_grouped_enc.values), num_workers=0)
    # GeM's p is scale-like: weight decay would drag it toward the degenerate p=0,
    # so it gets its own group with decay off. Empty for every other rung.
    decay = [q for n, q in model.named_parameters() if not n.endswith('avgpool.p')]
    nodecay = [q for n, q in model.named_parameters() if n.endswith('avgpool.p')]
    groups = [dict(params=decay, weight_decay=1e-4)]
    if nodecay:
        groups.append(dict(params=nodecay, weight_decay=0.0))
    opt = torch.optim.AdamW(groups, lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=2)

    progress = REPO / "experiments" / f"progress_{tag}.log"
    def note(line):
        print(line, flush=True)
        with open(progress, "a") as fh:
            fh.write(line + "\n"); fh.flush()

    note(f"=== training: stem={args.stem}  params={nparam/1e6:.2f}M  epochs<={args.epochs} "
         f"device={DEVICE} ===")
    best, best_state, bad, hist = -1, None, 0, []
    t_start = time.time()
    for ep in range(args.epochs):
        t0 = time.time(); model.train(); tot = n = 0; acts = []; c_losses = []; c_accs = []; s_losses = []
        for imgs, labs, aux in tqdm(loader, desc=f"ep{ep+1}", leave=False):
            imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
            emb = model(imgs)
            if args.head == 'colour':
                sem_b, col_b = ColourEmbeddingNet.blocks(emb)
                sem_loss, act = batch_hard_triplet_loss(sem_b, labs)
                cols = aux[:, 0].to(DEVICE)
                if args.colour_loss == 'ce':
                    logits = model.colour_logits(col_b)
                    c_loss = F.cross_entropy(logits, cols)
                    c_accs.append(float((logits.argmax(1) == cols).float().mean()))
                else:
                    c_loss, _ = batch_hard_triplet_loss(col_b, cols)
                loss = sem_loss + args.colour_weight * c_loss
                if args.arcface:                      # rung 6
                    loss = loss + args.arc_weight * F.cross_entropy(model.arc(sem_b, labs), labs)
                s_losses.append(sem_loss.item()); c_losses.append(c_loss.item())
            elif args.aux or args.arcface:
                sem_loss, act = batch_hard_triplet_loss(emb, labs)
                extra_loss = 0.0
                if args.aux:
                    aux = aux.to(DEVICE)
                    ce = [F.cross_entropy(h(emb), aux[:, j]) for j, h in enumerate(model.aux_heads)]
                    extra_loss = extra_loss + args.aux_weight * torch.stack(ce).mean()
                if args.arcface:
                    extra_loss = extra_loss + args.arc_weight * F.cross_entropy(model.arc(emb, labs), labs)
                loss = sem_loss + extra_loss
                s_losses.append(sem_loss.item()); c_losses.append(float(extra_loss))
            else:
                loss, act = batch_hard_triplet_loss(emb, labs)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * imgs.size(0); n += imgs.size(0); acts.append(act)
        t_train = time.time() - t0
        t1 = time.time(); m10 = probe_map10(model); t_eval = time.time() - t1
        hist.append(dict(epoch=ep+1, loss=tot/n, active=float(np.mean(acts)), probe_mAP10=m10,
                         t_train=round(t_train,1), t_eval=round(t_eval,1)))
        extra = ""
        # sem_loss sitting on MARGIN (0.20) means the semantic block has collapsed.
        if s_losses: extra += f" sem_loss {np.mean(s_losses):.4f}"
        if c_losses: extra += f" col_loss {np.mean(c_losses):.4f}"
        if c_accs: extra += f" col_acc {np.mean(c_accs):.3f}"
        if args.pool == 'gem':
            gp = [q for nm, q in model.named_parameters() if nm.endswith('avgpool.p')]
            if gp: extra += f" gem_p {float(gp[0]):.3f}"
        note(f"ep {ep+1}/{args.epochs} loss {tot/n:.4f} active {np.mean(acts):.1%}{extra} "
             f"probe_mAP@10 {m10:.4f} | train {t_train:.0f}s eval {t_eval:.0f}s "
             f"elapsed {(time.time()-t_start)/60:.1f}min lr {opt.param_groups[0]['lr']:.2e}")
        sched.step(m10)
        # Rolling checkpoint of the best state so far -- a run stopped for any reason
        # still leaves a usable model behind.
        if m10 > best:
            torch.save(model.state_dict(), REPO / "experiments" / f"ckpt_{tag}.pt")
        if m10 > best:
            best, best_state, bad = m10, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                note(f"early stopping at epoch {ep+1} (best probe mAP@10 {best:.4f})"); break
        stop_file = REPO / "experiments" / f"STOP_{tag}"
        if stop_file.exists():
            note(f"stop file seen -- finalising at epoch {ep+1} (best probe mAP@10 {best:.4f})")
            stop_file.unlink()
            break
    model.load_state_dict(best_state)
    wall = time.time() - t_start

    res = full_eval(model)
    res.update(stem=args.stem, pool=args.pool, head=args.head,
               aux=args.aux, arcface=args.arcface,
               colour_loss=args.colour_loss if args.head == 'colour' else None,
               colour_weight=args.colour_weight if args.head == 'colour' else None,
               emb_dim=model.out_dim,
               gem_p=next((float(q) for nm, q in model.named_parameters()
                           if nm.endswith('avgpool.p')), None),
               params_M=round(nparam/1e6, 2), epochs_run=len(hist),
               best_probe=best, wall_seconds=round(wall), history=hist)
    outp = REPO / "experiments" / f"result_{tag}.json"
    outp.write_text(json.dumps(res, indent=2))
    torch.save(model.state_dict(), REPO / "experiments" / f"model_{tag}.pt")
    print(f"\n=== {tag} ===")
    print(f"mAP@10 {res['mAP10']:.4f} (CI {res['ci_low']:.3f}-{res['ci_high']:.3f})  "
          f"strict {res['mAP10_strict']:.4f}  P@1 {res['P@1']:.4f} P@20 {res['P@20']:.4f}")
    print(f"epochs {res['epochs_run']}, wall {wall/60:.1f} min -> {outp}")

if __name__ == "__main__":
    main()
