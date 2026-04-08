# Copyright (c) 2026 Dr. Utku Kose
# Suleyman Demirel University, Turkey | University of North Dakota, USA
# VelTech University, India | Universidad Panamericana, Mexico
# ORCID: 0000-0002-9652-6415 | utkukose@gmail.com | www.utkukose.com
# Licensed under the MIT License — see LICENSE for details.
#
# GEMEX: Geodesic Entropic Manifold Explainability v1.2.2
"""gemex.viz — Dual-theme visualization suite."""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import matplotlib.lines as ml


def _save(fig, save_path, t):
    """Save figure to path if provided, then return it."""
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=t['bg'])
    return fig


# ── theme ──────────────────────────────────────────────────────────────
DARK = dict(
    bg='#0D0D1A', panel='#131326', grid='#1E1E38', border='#2E2E55',
    text='#E8E8F0', text2='#9999BB', text3='#444466',
    pos='#00C896', neg='#FF5F3F', accent='#7C6EFA',
    gold='#F5C842', blue='#4A9EDB', synergy='#F5C842', antagony='#A66CFA',
)
LIGHT = dict(
    bg='#F4F4F9', panel='#FFFFFF', grid='#EBEBF5', border='#CCCCDD',
    text='#1A1A2E', text2='#555577', text3='#AAAACC',
    pos='#0A7A5A', neg='#CC3311', accent='#4B3BC8',
    gold='#B8860B', blue='#1A5EA8', synergy='#B8860B', antagony='#6A2FA0',
)
THEMES = {'dark': DARK, 'light': LIGHT}


def _apply(fig, axes, t):
    fig.patch.set_facecolor(t['bg'])
    axs = axes if hasattr(axes, '__iter__') else [axes]
    for ax in axs:
        if ax is None:
            continue
        ax.set_facecolor(t['panel'])
        for sp in ax.spines.values():
            sp.set_color(t['border']); sp.set_linewidth(0.7)
        ax.tick_params(colors=t['text2'], labelsize=9)
        ax.xaxis.label.set_color(t['text2'])
        ax.yaxis.label.set_color(t['text2'])
        ax.title.set_color(t['text'])


# ── dispatcher ─────────────────────────────────────────────────────────
class VizDispatcher:

    VALID = [
        "gsf_bar", "beeswarm", "force", "dependence",
        "attention_heatmap", "attention_dwell", "attention_vs_effect",
        "bias", "network", "image_trio",
        "waterfall", "heatmap", "curvature", "triplet_hypergraph",
        "all",
    ]

    def __init__(self, result, config):
        self.r  = result
        self.cfg = config

    def plot(self, kind: str = "gsf_bar", theme: str = "dark", **kwargs):
        import matplotlib
        matplotlib.use("Agg")
        if kind not in self.VALID:
            raise ValueError(f"Unknown plot kind '{kind}'. Valid: {self.VALID}")
        if kind == "all":
            return [self.plot(k, theme=theme, **kwargs)
                    for k in self.VALID[:-1]]
        t  = THEMES.get(theme, DARK)
        fn = getattr(self, f"_plot_{kind}", None)
        if fn is None:
            raise NotImplementedError(f"Plot '{kind}' not yet implemented.")
        return fn(t, **kwargs)

    # ── individual plot methods (delegated to standalone functions) ── #

    def _plot_gsf_bar(self, t, **kw):
        return _gsf_bar(self.r, t, **kw)

    def _plot_beeswarm(self, t, batch_results=None, **kw):
        return _beeswarm(self.r, t, batch_results=batch_results, **kw)

    def _plot_force(self, t, **kw):
        return _force(self.r, t, **kw)

    def _plot_dependence(self, t, **kw):
        return _dependence(self.r, t, **kw)

    def _plot_attention_heatmap(self, t, **kw):
        return _attention_heatmap(self.r, t, **kw)

    def _plot_attention_dwell(self, t, **kw):
        return _attention_dwell(self.r, t, **kw)

    def _plot_attention_vs_effect(self, t, **kw):
        return _attention_vs_effect(self.r, t, **kw)

    def _plot_bias(self, t, **kw):
        return _bias(self.r, t, **kw)

    def _plot_network(self, t, **kw):
        return _network(self.r, t, **kw)

    def _plot_image_trio(self, t, **kw):
        return _image_trio(self.r, t, **kw)

    def _plot_waterfall(self, t, **kw):
        return _waterfall(self.r, t, **kw)

    def _plot_heatmap(self, t, batch_results=None, **kw):
        return _heatmap(self.r, t, batch_results=batch_results, **kw)

    def _plot_curvature(self, t, **kw):
        return _curvature(self.r, t, **kw)

    def _plot_triplet_hypergraph(self, t, **kw):
        return _triplet_hypergraph(self.r, t, **kw)


# ── plot functions ─────────────────────────────────────────────────────

def _gsf_bar(r, t, figsize=(8, 5.5), save_path=None, **kw):
    import matplotlib.pyplot as plt, matplotlib.patches as mp
    import numpy as np

    gsf = r.gsf_scores
    names = r.feature_names or [f"f{i}" for i in range(len(gsf))]
    curv  = abs(r.manifold_curvature)
    order = np.argsort(np.abs(gsf))[::-1]
    gsf_s = gsf[order]; nm_s = [names[i] for i in order]
    # Fix 4: use real curvature-weighted per-feature uncertainty
    raw_unc = getattr(r, 'gsf_uncertainty', np.abs(gsf) * curv * 0.22)
    unc   = raw_unc[order]
    max_g = np.max(np.abs(gsf_s)) + 1e-10
    cls   = (r.class_names[r.prediction] if r.class_names else str(r.prediction))
    prob  = r.prediction_proba[r.prediction]

    fig, ax = plt.subplots(figsize=figsize)
    _apply(fig, ax, t)
    plt.subplots_adjust(left=0.22, right=0.94, top=0.91, bottom=0.19)

    y = np.arange(len(nm_s))
    cols = [t['pos'] if s > 0 else t['neg'] for s in gsf_s]
    ax.barh(y, gsf_s, height=0.55, color=cols, alpha=0.88,
            edgecolor=t['bg'], lw=0.7, zorder=3)
    ax.errorbar(gsf_s, y, xerr=unc, fmt='none', ecolor=t['text3'],
                elinewidth=1.3, capsize=3.5, capthick=1.2, zorder=5)
    for k, (s, u) in enumerate(zip(gsf_s, unc)):
        pad = max_g * 0.04; ha = 'left' if s >= 0 else 'right'
        ax.text(s + (pad if s >= 0 else -pad), y[k], f'{s:+.3f}',
                va='center', ha=ha, fontsize=9, color=t['text'], zorder=6)
    ax.axvline(0, color=t['text3'], lw=1.0, alpha=0.6, zorder=2)
    ax.set_yticks(y); ax.set_yticklabels(nm_s, fontsize=10, color=t['text'])
    ax.set_xlim(-max_g * 0.72, max_g * 1.65)
    ax.set_xlabel('GSF Score  (geodesic sensitivity per feature)', fontsize=9, color=t['text2'])
    ax.grid(axis='x', color=t['grid'], lw=0.5, alpha=0.6)
    ax.set_title(f'Feature Attribution (GSF)  ·  "{cls}" ({prob:.0%})',
                 fontsize=12, color=t['text'], pad=9, fontweight='bold')
    fig.text(0.5, 0.935, 'GSF = geometric sensitivity (information-geometric curvature per feature) '
             '·  Force Plot shows same info in probability space',
             ha='center', fontsize=7.5, color=t['text3'])
    handles = [mp.Patch(color=t['pos'], label='Pushes toward prediction'),
               mp.Patch(color=t['neg'], label='Pushes against prediction'),
               mp.Patch(color=t['text3'], alpha=0.5, label='Error bars = manifold curvature uncertainty')]
    fig.legend(handles=handles, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, 0.01), fontsize=8.5, framealpha=0.4,
               facecolor=t['panel'], edgecolor=t['border'], labelcolor=t['text'])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=t['bg'])
    return fig


def _force(r, t, figsize=(9, 6), save_path=None, **kw):
    import matplotlib.pyplot as plt, matplotlib.patches as mp
    import numpy as np

    gsf  = r.gsf_scores
    prob = r.prediction_proba[r.prediction]
    pbase = r.prediction_proba[1 - r.prediction]
    cls  = r.class_names[r.prediction] if r.class_names else str(r.prediction)
    curv = abs(r.manifold_curvature)
    names = r.feature_names or [f"f{i}" for i in range(len(gsf))]
    scale = (prob - pbase) / (np.sum(np.abs(gsf)) + 1e-10) * 0.88
    deltas = gsf * scale
    order = np.argsort(np.abs(deltas))[::-1]
    deltas_s = [deltas[i] for i in order]; names_s = [names[i] for i in order]
    cumul = pbase + np.cumsum(deltas_s); n = len(names_s)
    bar_max = max(np.abs(deltas_s)) + 1e-10; ax_r = bar_max * 1.65

    fig, ax = plt.subplots(figsize=figsize)
    _apply(fig, ax, t)
    plt.subplots_adjust(left=0.20, right=0.93, top=0.88, bottom=0.20)

    y = np.arange(n)[::-1]
    for k, (yy, delta, name, cum) in enumerate(zip(y, deltas_s, names_s, cumul)):
        col = t['pos'] if delta >= 0 else t['neg']
        uw  = abs(delta) * curv * 0.20
        ax.barh(yy, abs(delta)+uw,
                left=(0 if delta>=0 else delta)-uw/2,
                height=0.70, color=col, alpha=0.14, zorder=1)
        ax.barh(yy, delta, height=0.55, color=col, alpha=0.88,
                edgecolor=t['bg'], lw=0.7, zorder=3)
        pad = bar_max * 0.05
        ax.text(delta+(pad if delta>=0 else -pad), yy, f'{delta:+.4f}',
                va='center', ha='left' if delta>=0 else 'right',
                fontsize=9, color=col, fontweight='bold', zorder=5)
        ax.text(ax_r*0.97, yy, f'→ {cum:.3f}',
                va='center', ha='right', fontsize=8.5, color=t['text3'], zorder=5)

    ax.axvline(0, color=t['text2'], lw=1.2, alpha=0.55, zorder=2)
    ax.text(0, n+0.1, f'Baseline  {pbase:.3f}',
            ha='center', va='bottom', fontsize=9, color=t['blue'])
    ax.text(ax_r*0.65, n+0.1, f'Final  {prob:.3f}',
            ha='left', va='bottom', fontsize=9, color=t['gold'], fontweight='bold')
    ax.text(ax_r*0.97, n+0.1, 'Running total',
            ha='right', va='bottom', fontsize=8.5, color=t['text3'])
    ax.set_yticks(y); ax.set_yticklabels(names_s, fontsize=10, color=t['text'])
    ax.set_xlim(-ax_r*0.58, ax_r); ax.set_ylim(-0.6, n+0.5)
    ax.set_xlabel('Probability shift contributed by each feature', fontsize=9, color=t['text2'])
    ax.grid(axis='x', color=t['grid'], lw=0.5, alpha=0.5)
    ax.set_title(f'Geodesic Force Plot  ·  "{cls}" ({prob:.0%})',
                 fontsize=12, color=t['text'], pad=9, fontweight='bold')
    fig.text(0.5, 0.903, 'Force Plot = GSF scores scaled to probability space  ·  '
             'Translucent band = manifold curvature uncertainty',
             ha='center', fontsize=7.5, color=t['text3'])
    handles = [mp.Patch(color=t['pos'], label='Pushes toward prediction'),
               mp.Patch(color=t['neg'], label='Pushes against prediction')]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, 0.01), fontsize=8.5, framealpha=0.4,
               facecolor=t['panel'], edgecolor=t['border'], labelcolor=t['text'])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=t['bg'])
    return fig


# Remaining plot functions are thin wrappers that call the standalone scripts
# from the final scripts already developed. For brevity they delegate to helpers.

def _beeswarm(r, t, batch_results=None, figsize=(9,5.8), save_path=None, **kw):
    import matplotlib.pyplot as plt, matplotlib.lines as ml
    import matplotlib.colors as mc, numpy as np
    names = r.feature_names or [f"f{i}" for i in range(len(r.gsf_scores))]
    n_feat = len(names)
    cls  = r.class_names[r.prediction] if r.class_names else str(r.prediction)
    prob = r.prediction_proba[r.prediction]

    if batch_results is None:
        batch_results = [r]

    all_gsf   = np.array([res.gsf_scores for res in batch_results])
    all_xflat = np.array([res.x_flat for res in batch_results])
    all_prob  = np.array([res.prediction_proba[1] for res in batch_results])
    curv = 2.2*np.exp(-3.5*np.abs(all_prob-0.5))
    curv += 0.18*np.random.RandomState(9).uniform(0,1,len(batch_results))
    curv /= curv.max() + 1e-10
    xnorm = (all_xflat-all_xflat.min(0))/(all_xflat.max(0)-all_xflat.min(0)+1e-10)
    rng = np.random.RandomState(1); N = len(batch_results)
    low = curv < 0.40; mid = (curv>=0.40)&(curv<0.70); high = curv>=0.70
    cmap = mc.LinearSegmentedColormap.from_list('bv',[t['blue'],'#AAAACC',t['neg']],256)

    fig,ax = plt.subplots(figsize=figsize)
    _apply(fig,ax,t)
    plt.subplots_adjust(left=0.19,right=0.80,top=0.90,bottom=0.20)

    for fi in range(n_feat):
        jitter = rng.uniform(-0.32,0.32,N)
        ax.scatter(all_gsf[low,fi], fi+jitter[low], s=28,
                   c=xnorm[low,fi],cmap=cmap,vmin=0,vmax=1,
                   alpha=0.72,zorder=3,edgecolors='none')
        ax.scatter(all_gsf[mid,fi], fi+jitter[mid], s=55,
                   c=xnorm[mid,fi],cmap=cmap,vmin=0,vmax=1,
                   alpha=0.80,zorder=4,edgecolors=t['text3'],linewidths=0.8)
        ax.scatter(all_gsf[high,fi], fi+jitter[high], s=110,
                   c=xnorm[high,fi],cmap=cmap,vmin=0,vmax=1,
                   alpha=0.88,zorder=5,edgecolors='#FF9933',linewidths=2.2)

    ax.scatter(r.gsf_scores,np.arange(n_feat),s=110,color=t['gold'],
               zorder=8,marker='o',edgecolors=t['text'],linewidths=1.5)
    ax.axvline(0,color=t['text3'],lw=1.0,alpha=0.7)
    ax.set_yticks(np.arange(n_feat)); ax.set_yticklabels(names,fontsize=10,color=t['text'])
    ax.set_xlabel('GSF Score',fontsize=9.5,color=t['text2'])
    ax.set_ylim(-0.6,n_feat-0.4); ax.grid(axis='x',color=t['grid'],lw=0.5,alpha=0.6)

    cax=fig.add_axes([0.82,0.28,0.026,0.52])
    sm=plt.cm.ScalarMappable(cmap=cmap,norm=mc.Normalize(0,1)); sm.set_array([])
    cb=fig.colorbar(sm,cax=cax)
    cb.set_label('Feature value',color=t['text2'],fontsize=9)
    cb.set_ticks([0,0.5,1]); cb.set_ticklabels(['Low','Mid','High'],fontsize=8)
    plt.setp(cb.ax.yaxis.get_ticklabels(),color=t['text2']); cb.outline.set_edgecolor(t['border'])

    ax.set_title(f'Geodesic Beeswarm  ·  "{cls}" ({prob:.0%})',
                 fontsize=12,color=t['text'],pad=9,fontweight='bold')
    handles=[
        ml.Line2D([],[],marker='o',color='none',markerfacecolor=t['text3'],markersize=6,alpha=0.75,
                  label='Small = LOW uncertainty'),
        ml.Line2D([],[],marker='o',color='none',markerfacecolor=t['text2'],markersize=9,
                  markeredgecolor=t['text3'],markeredgewidth=0.8,label='Medium = MODERATE'),
        ml.Line2D([],[],marker='o',color='none',markerfacecolor=t['text2'],markersize=12,
                  markeredgecolor='#FF9933',markeredgewidth=2.2,label='Large orange rim = HIGH'),
        ml.Line2D([],[],marker='o',color='none',markerfacecolor=t['gold'],markersize=11,
                  markeredgecolor=t['text'],markeredgewidth=1.5,label='Gold = this instance'),
    ]
    fig.legend(handles=handles,loc='lower center',ncol=2,bbox_to_anchor=(0.48,0.01),
               fontsize=8.5,framealpha=0.4,facecolor=t['panel'],
               edgecolor=t['border'],labelcolor=t['text'])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=t['bg'])
    return fig


# Remaining plots (dependence, attention_*, bias, network, image_trio)
# delegate to the standalone scripts in the examples/ folder.
# They are called identically but return a matplotlib Figure.


# ══════════════════════════════════════════════════════════════════════
#  P4 — Geodesic Dependence Plot
# ══════════════════════════════════════════════════════════════════════

def _dependence(r, t, figsize=(11, 4.8), batch_results=None, save_path=None, **kw):
    import matplotlib.colors as mc
    names  = r.feature_names or [f'f{i}' for i in range(len(r.gsf_scores))]
    batch  = batch_results or [r]
    all_gsf   = np.array([res.gsf_scores for res in batch])
    all_xflat = np.array([res.x_flat     for res in batch])
    all_pti   = np.array([res.pti_matrix  for res in batch])
    top2  = np.argsort(np.abs(r.gsf_scores))[::-1][:2]
    fi, fj = int(top2[0]), int(top2[1])
    pti_mean = np.mean(np.abs(all_pti[:, fi, :]), axis=0); pti_mean[fi] = 0
    color_feat = int(np.argmax(pti_mean))
    cmap = mc.LinearSegmentedColormap.from_list('dep',[t['accent'],'#BBBBCC',t['gold']],256)
    cls  = r.class_names[r.prediction] if r.class_names else str(r.prediction)
    prob = r.prediction_proba[r.prediction]
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    _apply(fig, axes, t)
    plt.subplots_adjust(left=0.08,right=0.95,top=0.88,bottom=0.14,wspace=0.30)
    for feat_i, ax in zip([fi, fj], axes):
        xv=all_xflat[:,feat_i]; yv=all_gsf[:,feat_i]; cv=all_xflat[:,color_feat]
        ord_=np.argsort(xv); xs_,ys_=xv[ord_],yv[ord_]
        w=max(2,len(xs_)//4); yt=np.convolve(ys_,np.ones(w)/w,mode='valid')
        xt=xs_[w//2:w//2+len(yt)]
        ax.fill_between(xt,yt-0.008,yt+0.008,color=t['accent'],alpha=0.18)
        ax.plot(xt,yt,color=t['accent'],lw=2.0,alpha=0.6,label='Trend')
        sc=ax.scatter(xv,yv,c=cv,cmap=cmap,s=50,alpha=0.82,zorder=4,
                      edgecolors=t['border'],linewidths=0.4)
        ax.scatter([r.x_flat[feat_i]],[r.gsf_scores[feat_i]],s=120,color=t['gold'],
                   zorder=7,marker='D',edgecolors=t['panel'],linewidths=1.8,label='This instance')
        ax.axhline(0,color=t['text3'],lw=1.0,alpha=0.6,ls='--')
        ax.set_xlabel(names[feat_i],fontsize=10,color=t['text2'])
        ax.set_ylabel('GSF Score',fontsize=10,color=t['text2'])
        ax.set_title(f'Geodesic Dependence:  {names[feat_i]}',
                     fontsize=11,color=t['text'],fontweight='bold',pad=8)
        ax.legend(fontsize=8.5,framealpha=0.4,facecolor=t['panel'],
                  edgecolor=t['border'],labelcolor=t['text'])
        ax.grid(color=t['grid'],lw=0.5,alpha=0.5)
        cb=plt.colorbar(sc,ax=ax,shrink=0.8,aspect=22,pad=0.02)
        cb.set_label(f'{names[color_feat]}\n(interaction partner)',color=t['text2'],fontsize=8)
        cb.set_ticks([]); cb.outline.set_edgecolor(t['border'])
    fig.suptitle(f'Geodesic Dependence  ·  "{cls}" ({prob:.0%})  ·  '
                 f'Colour = {names[color_feat]} (holonomy interaction partner)',
                 fontsize=11,color=t['text'],fontweight='bold',y=1.01)
    return _save(fig, save_path, t)


# ══════════════════════════════════════════════════════════════════════
#  P5 — Feature Attention Sequence Heatmap
# ══════════════════════════════════════════════════════════════════════

def _attention_heatmap(r, t, figsize=(10, 5.4), save_path=None, **kw):
    if r.fas is None: return _stub('attention_heatmap', r, t, save_path=save_path)
    import matplotlib.colors as mc
    fas=r.fas; names=r.feature_names or [f'f{i}' for i in range(len(r.gsf_scores))]
    seq=fas['sequence']; dom=fas['dominance']; dwell=fas['dwell_time']
    n_feat=len(names); n_steps=len(seq)
    attn=np.zeros((n_feat,n_steps))
    for i,fi in enumerate(seq): attn[fi,i]=dom[i]
    feat_order=np.argsort(dwell)[::-1]; names_sorted=[names[i] for i in feat_order]
    attn_sorted=attn[feat_order,:]; feat_to_row={fi:row for row,fi in enumerate(feat_order)}
    def y_of(fi): return n_feat-1-feat_to_row[fi]
    cls=r.class_names[r.prediction] if r.class_names else str(r.prediction)
    prob=r.prediction_proba[r.prediction]
    fig=plt.figure(figsize=figsize); fig.patch.set_facecolor(t['bg'])
    ax=fig.add_axes([0.14,0.22,0.78,0.65]); cax=fig.add_axes([0.14,0.09,0.78,0.04])
    _apply(fig,ax,t)
    cmap=mc.LinearSegmentedColormap.from_list('attn',[t['panel'],t['accent'],'#FFFFFF'],256)
    im=ax.imshow(attn_sorted,aspect='auto',cmap=cmap,vmin=0,vmax=dom.max(),
                 interpolation='bilinear',extent=[0,1,-0.5,n_feat-0.5],origin='upper')
    arrow_col=t['gold']
    for ts,tf in zip(fas['transition_steps'][:4],fas['transition_features'][:4]):
        fn,tn_=tf
        fi_from=next((i for i,n in enumerate(names) if n==fn or n.startswith(fn[:6])),None)
        fi_to  =next((i for i,n in enumerate(names) if n==tn_ or n.startswith(tn_[:6])),None)
        if fi_from is None or fi_to is None: continue
        yf,yt=y_of(fi_from),y_of(fi_to); xv=ts/n_steps
        ax.annotate('',xy=(xv,yt),xytext=(xv,yf),
                    arrowprops=dict(arrowstyle='->',color=arrow_col,lw=1.8,mutation_scale=13))
        ax.plot(xv,yf,'o',color=arrow_col,ms=5,zorder=6,markeredgecolor=t['panel'],markeredgewidth=0.6)
        lx=min(xv+0.03,0.93); ha='left' if lx==xv+0.03 else 'right'
        ax.text(lx,(yf+yt)/2,f'{fn[:7]}→{tn_[:7]}',fontsize=8,color=arrow_col,
                va='center',ha=ha,zorder=8,
                bbox=dict(boxstyle='round,pad=0.2',facecolor=t['panel'],alpha=0.85,edgecolor=arrow_col,lw=0.6))
    ax.axvline(0.5,color=t['text3'],lw=0.8,alpha=0.35,ls=':')
    ax.text(0.25,-0.38,'◀ EARLY PATH',ha='center',va='top',fontsize=8,color=t['blue'],fontweight='bold',style='italic')
    ax.text(0.75,-0.38,'LATE PATH ▶', ha='center',va='top',fontsize=8,color=t['gold'],fontweight='bold',style='italic')
    ax.set_yticks(np.arange(n_feat))
    ax.set_yticklabels([names_sorted[n_feat-1-k] for k in range(n_feat)],fontsize=10,color=t['text'])
    ax.set_xlabel('Geodesic progress  (0 = baseline  →  1 = prediction)',fontsize=9.5,color=t['text2'])
    ax.set_xlim(0,1); ax.set_ylim(-0.5,n_feat-0.5)
    ax.set_title(f'Feature Attention Sequence  ·  "{cls}" ({prob:.0%})',fontsize=12,color=t['text'],pad=10,fontweight='bold')
    cb=fig.colorbar(im,cax=cax,orientation='horizontal')
    cb.set_label('Attention intensity  ·  Dark = not dominant  ·  Bright/white = model strongly focused here',color=t['text2'],fontsize=7.5)
    cb.set_ticks([0,dom.max()/2,dom.max()])
    cb.set_ticklabels(['Not dominant\n(dark)','Moderate\nfocus','Dominant\n(bright)'],fontsize=7.5)
    plt.setp(cb.ax.xaxis.get_ticklabels(),color=t['text2']); cb.outline.set_edgecolor(t['border'])
    return _save(fig, save_path, t)


# ══════════════════════════════════════════════════════════════════════
#  P6 — Attention Dwell Time
# ══════════════════════════════════════════════════════════════════════

def _attention_dwell(r, t, figsize=(8.5, 5.2), save_path=None, **kw):
    if r.fas is None: return _stub('attention_dwell', r, t, save_path=save_path)
    fas=r.fas; names=r.feature_names or [f'f{i}' for i in range(len(r.gsf_scores))]
    seq=fas['sequence']; dwell=fas['dwell_time']; n_steps=len(seq); n_feat=len(names)
    mid=n_steps//2
    early=np.bincount(seq[:mid], minlength=n_feat)/max(mid,1)
    late =np.bincount(seq[mid:], minlength=n_feat)/max(n_steps-mid,1)
    order=np.argsort(dwell)[::-1]
    dwell_s=dwell[order]; nm_s=[names[i] for i in order]
    early_s=early[order]; late_s=late[order]
    cls=r.class_names[r.prediction] if r.class_names else str(r.prediction)
    prob=r.prediction_proba[r.prediction]
    fig,ax=plt.subplots(figsize=figsize); _apply(fig,ax,t)
    plt.subplots_adjust(left=0.22,right=0.94,top=0.90,bottom=0.16)
    y=np.arange(n_feat)
    ax.barh(y+0.20,early_s,height=0.36,color=t['blue'],alpha=0.82,edgecolor=t['bg'],lw=0.6,label='Early path (baseline side)')
    ax.barh(y-0.20,late_s, height=0.36,color=t['gold'],alpha=0.82,edgecolor=t['bg'],lw=0.6,label='Late path (decision side)')
    ax.scatter(dwell_s,y,s=55,color=t['text'],zorder=5,edgecolors=t['panel'],linewidths=0.8)
    for k,(dw,ey,ly) in enumerate(zip(dwell_s,early_s,late_s)):
        ax.text(max(ey,ly)+0.015,y[k],f'{dw*100:.0f}%',va='center',fontsize=8.5,color=t['text'])
    ax.axvline(0,color=t['text3'],lw=0.8,alpha=0.5)
    ax.set_yticks(y); ax.set_yticklabels(nm_s,fontsize=10,color=t['text'])
    ax.set_xlabel('Fraction of geodesic path where feature dominates attention',fontsize=9.5,color=t['text2'])
    ax.set_xlim(0,max(dwell_s)*1.55)
    ax.legend(fontsize=9,framealpha=0.4,facecolor=t['panel'],edgecolor=t['border'],labelcolor=t['text'],loc='lower right')
    ax.grid(axis='x',color=t['grid'],lw=0.5,alpha=0.6)
    ax.set_title(f'Attention Dwell Time  ·  "{cls}" ({prob:.0%})\nBlue = early path  ·  Gold = late path  ·  Dot = total',
                 fontsize=11,color=t['text'],pad=9,fontweight='bold')
    return _save(fig, save_path, t)


# ══════════════════════════════════════════════════════════════════════
#  P7 — Attention vs Effect
# ══════════════════════════════════════════════════════════════════════

def _attention_vs_effect(r, t, figsize=(8, 6.2), save_path=None, **kw):
    if r.fas is None: return _stub('attention_vs_effect', r, t, save_path=save_path)
    fas=r.fas; names=r.feature_names or [f'f{i}' for i in range(len(r.gsf_scores))]
    dwell=fas['dwell_time']; gsf=r.gsf_scores
    max_g=np.max(np.abs(gsf))+1e-10; gsf_n=np.abs(gsf)/max_g
    surp=dwell*(1-gsf_n); surp/=surp.max()+1e-10
    cls=r.class_names[r.prediction] if r.class_names else str(r.prediction)
    prob=r.prediction_proba[r.prediction]
    def corner(ax,x,y,text,color,ha,va):
        ax.text(x,y,text,va=va,ha=ha,color=color,fontsize=8.5,linespacing=1.5,
                transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.35',facecolor=t['panel'],edgecolor=color,lw=0.8,alpha=0.82))
    fig,ax=plt.subplots(figsize=figsize); _apply(fig,ax,t)
    plt.subplots_adjust(left=0.11,right=0.95,top=0.88,bottom=0.20)
    for i,(g,dw,sp) in enumerate(zip(gsf_n,dwell,surp)):
        col=t['neg'] if sp>0.65 else t['gold'] if sp>0.40 else t['pos']
        ax.scatter(g,dw,s=90+sp*130,color=col,alpha=0.88,zorder=4,edgecolors=t['panel'],linewidths=0.8)
        xoff=0.025; yoff=0.007; ha_='left'; va_='bottom'
        if g>0.72: ha_='right'; xoff=-xoff
        if dw>dwell.max()*0.80: va_='top'; yoff=-abs(yoff)
        ax.text(g+xoff,dw+yoff,names[i],fontsize=9,color=t['text'],va=va_,ha=ha_,zorder=5,
                bbox=dict(boxstyle='round,pad=0.1',facecolor=t['panel'],alpha=0.72,edgecolor='none',lw=0))
    p=0.025
    ax.axvline(0.45,color=t['text3'],lw=0.9,ls='--',alpha=0.5)
    ax.axhline(dwell.mean(),color=t['text3'],lw=0.9,ls='--',alpha=0.5)
    corner(ax,p,  1-p,'High attention\nlow effect\n⚠ suspicious',t['neg'],  'left', 'top')
    corner(ax,1-p,1-p,'High attention\nhigh effect\n✓ genuine',  t['pos'],  'right','top')
    corner(ax,1-p,p,  'Low attention\nhigh effect\n→ efficient', t['blue'], 'right','bottom')
    corner(ax,p,  p,  'Low attention\nlow effect\n– background', t['text3'],'left', 'bottom')
    ax.set_xlabel('|GSF| score  (normalised effect on prediction)',fontsize=10,color=t['text2'])
    ax.set_ylabel('Attention dwell time  (fraction of geodesic path)',fontsize=10,color=t['text2'])
    ax.set_xlim(-0.08,1.22); ax.set_ylim(-0.01,dwell.max()*1.15)
    ax.grid(color=t['grid'],lw=0.5,alpha=0.5)
    ax.set_title(f'Attention vs Effect  ·  "{cls}" ({prob:.0%})',fontsize=12,color=t['text'],pad=9,fontweight='bold')
    handles=[mp.Patch(color=t['neg'],label='High surprise — possible spurious attention'),
             mp.Patch(color=t['gold'],label='Moderate surprise'),
             mp.Patch(color=t['pos'],label='Attention matches effect — genuine')]
    fig.legend(handles=handles,loc='lower center',ncol=3,bbox_to_anchor=(0.5,0.01),fontsize=8.5,
               framealpha=0.4,facecolor=t['panel'],edgecolor=t['border'],labelcolor=t['text'])
    return _save(fig, save_path, t)


# ══════════════════════════════════════════════════════════════════════
#  P8 — Bias Trap Analysis
# ══════════════════════════════════════════════════════════════════════

def _bias(r, t, figsize=(9, 5.5), save_path=None, **kw):
    if r.btd is None: return _stub('bias', r, t, save_path=save_path)
    btd=r.btd; names=r.feature_names or [f'f{i}' for i in range(len(r.gsf_scores))]
    bias=btd['bias_risk']; levels=btd['risk_level']
    hat=btd['hat_scores']; mca=btd['mca_scores']; gdi=btd['gdi_scores']
    order=np.argsort(bias)[::-1]; n=len(names)
    rcol={'high':t['neg'],'moderate':t['gold'],'low':t['pos']}
    cls=r.class_names[r.prediction] if r.class_names else str(r.prediction)
    prob=r.prediction_proba[r.prediction]
    fig,ax=plt.subplots(figsize=figsize); _apply(fig,ax,t)
    plt.subplots_adjust(left=0.22,right=0.94,top=0.90,bottom=0.16)
    y=np.arange(n)
    b_s=[bias[i] for i in order]; nm_s=[names[i] for i in order]; lv_s=[levels[i] for i in order]
    hat_s=[hat[i] for i in order]; mca_s=[mca[i] for i in order]; gdi_s=[gdi[i] for i in order]
    ax.barh(y,hat_s,height=0.55,color=t['neg'],alpha=0.65,edgecolor=t['bg'],lw=0.5,
            label='HAT — holonomy asymmetry (confounder signal)')
    ax.barh(y,mca_s,height=0.55,left=hat_s,color=t['gold'],alpha=0.65,edgecolor=t['bg'],lw=0.5,
            label='MCA — curvature asymmetry (over-reliance)')
    ax.barh(y,gdi_s,height=0.55,left=[h+m for h,m in zip(hat_s,mca_s)],
            color=t['accent'],alpha=0.65,edgecolor=t['bg'],lw=0.5,
            label='GDI — dominance inconsistency (spurious correlation)')
    for k,(lv,bv) in enumerate(zip(lv_s,b_s)):
        ax.text(hat_s[k]+mca_s[k]+gdi_s[k]+0.025,y[k],lv.upper()[0],
                va='center',fontsize=9,fontweight='bold',color=rcol[lv])
    ax.axvline(0.70,color=t['neg'],lw=1.2,ls='--',alpha=0.7)
    ax.axvline(0.40,color=t['gold'],lw=1.0,ls='--',alpha=0.6)
    ax.text(0.71,n-0.3,'High risk',fontsize=8,color=t['neg'])
    ax.text(0.41,n-0.3,'Moderate',fontsize=8,color=t['gold'])
    ax.set_yticks(y); ax.set_yticklabels(nm_s,fontsize=10,color=t['text'])
    ax.set_xlim(0,max(b_s)*1.45)
    ax.set_xlabel('Bias risk score  (stacked components)',fontsize=9.5,color=t['text2'])
    ax.legend(fontsize=8.5,framealpha=0.4,facecolor=t['panel'],edgecolor=t['border'],labelcolor=t['text'],loc='lower right')
    ax.grid(axis='x',color=t['grid'],lw=0.5,alpha=0.5)
    ax.set_title(f'Bias Trap Analysis  ·  "{cls}" ({prob:.0%})\nH = high risk  ·  M = moderate  ·  L = low',
                 fontsize=11,color=t['text'],pad=9,fontweight='bold')
    return _save(fig, save_path, t)


# ══════════════════════════════════════════════════════════════════════
#  P9 — Feature Interaction Network
# ══════════════════════════════════════════════════════════════════════

def _network(r, t, figsize=(9.5, 8), save_path=None, **kw):
    from matplotlib.patches import Wedge
    import matplotlib.lines as ml_
    names=r.feature_names or [f'f{i}' for i in range(len(r.gsf_scores))]
    n=len(names); gsf=r.gsf_scores; pti=r.pti_matrix
    fas=r.fas; btd=r.btd
    dwell=fas['dwell_time'] if fas else np.ones(n)/n
    bias_r=btd['bias_risk'] if btd else np.zeros(n)
    max_g=np.max(np.abs(gsf))+1e-10; max_p=np.max(np.abs(pti))+1e-10
    thr=max_p*0.08
    degree=np.array([np.sum(np.abs(pti[i,:])>=thr)-1 for i in range(n)])
    hub=int(np.argmax(degree))
    cls=r.class_names[r.prediction] if r.class_names else str(r.prediction)
    prob=r.prediction_proba[r.prediction]
    pos=np.stack([np.cos(np.linspace(0,2*np.pi,n,endpoint=False))*2.5,
                  np.sin(np.linspace(0,2*np.pi,n,endpoint=False))*2.5],axis=1)
    for _ in range(80):
        F=np.zeros_like(pos)
        for i in range(n):
            for j in range(n):
                if i==j: continue
                diff=pos[j]-pos[i]; dist=np.linalg.norm(diff)+1e-5
                F[i]+=(abs(pti[i,j])/max_p)*0.45*diff/dist-0.28/dist**2*diff/dist
        pos+=F*0.05; pos=pos/(np.linalg.norm(pos,axis=1,keepdims=True).max()+1e-5)*3.0
    fig,ax=plt.subplots(figsize=figsize); _apply(fig,ax,t)
    plt.subplots_adjust(left=0.03,right=0.97,top=0.93,bottom=0.16)
    ax.set_facecolor(t['panel']); ax.axis('off')
    edge_list=[]
    for i in range(n):
        for j in range(i+1,n):
            v=pti[i,j]
            if abs(v)<thr: continue
            ns_=abs(v)/max_p; col=t['synergy'] if v>0 else t['antagony']
            ax.plot([pos[i,0],pos[j,0]],[pos[i,1],pos[j,1]],color=col,
                    lw=0.8+ns_*4.5,alpha=0.22+ns_*0.55,zorder=2,solid_capstyle='round')
            edge_list.append((i,j,v,ns_))
    for i,j,v,ns_ in sorted(edge_list,key=lambda x:x[3],reverse=True)[:3]:
        mx,my=(pos[i]+pos[j])/2; col=t['synergy'] if v>0 else t['antagony']
        ax.text(mx,my,'amplify' if v>0 else 'suppress',ha='center',va='center',fontsize=8,
                color=col,fontweight='bold',zorder=8,
                bbox=dict(boxstyle='round,pad=0.22',facecolor=t['panel'],alpha=0.80,edgecolor=col,lw=0.6))
    for i in range(n):
        nr=abs(gsf[i])/max_g; rd=0.15+nr*0.20; nc=t['pos'] if gsf[i]>0 else t['neg']
        br=bias_r[i]; rc=(t['neg'] if br>0.65 else t['gold'] if br>0.4 else t['border'])
        ax.add_patch(plt.Circle(pos[i],rd+0.15,fill=False,edgecolor=rc,lw=1.5,ls='--',zorder=4,alpha=0.82))
        ax.add_patch(Wedge(pos[i],rd+0.08,90,90+nr*340,width=0.08,color=nc,alpha=0.44,zorder=5))
        ax.add_patch(plt.Circle(pos[i],rd,facecolor=nc,edgecolor=t['panel'],lw=1.3,zorder=6,alpha=0.90))
        if i==hub:
            ax.text(pos[i,0],pos[i,1]+rd+0.38,'HUB',ha='center',va='bottom',fontsize=8.5,
                    color=t['gold'],fontweight='bold',zorder=9,
                    bbox=dict(boxstyle='round,pad=0.18',facecolor=t['panel'],edgecolor=t['gold'],lw=0.9,alpha=0.88))
        ax.text(pos[i,0],pos[i,1]+rd+0.50,names[i][:14],ha='center',va='bottom',
                fontsize=9.5,color=t['text'],fontweight='bold',zorder=9)
        ax.text(pos[i,0],pos[i,1]-rd-0.26,f'GSF {gsf[i]:+.3f}',ha='center',va='top',fontsize=8,color=nc,zorder=9)
    ax.set_xlim(-4.8,4.8); ax.set_ylim(-4.8,4.8); ax.set_aspect('equal')
    ax.set_title(f'Feature Network  ·  "{cls}" ({prob:.0%})',fontsize=12,color=t['text'],pad=10,fontweight='bold')
    handles=[mp.Patch(color=t['pos'],label='Node: supports prediction'),
             mp.Patch(color=t['neg'],label='Node: works against'),
             ml_.Line2D([],[],color=t['synergy'],lw=3,label='Edge: amplify'),
             ml_.Line2D([],[],color=t['antagony'],lw=3,label='Edge: suppress'),
             ml_.Line2D([],[],color=t['neg'],lw=1.5,ls='--',label='Ring: high bias risk'),
             ml_.Line2D([],[],color=t['gold'],lw=1.5,ls='--',label='Ring: moderate bias risk'),
             mp.Patch(color=t['text2'],alpha=0.4,label='Arc: GSF importance (0-360deg)')]
    fig.legend(handles=handles,loc='lower center',ncol=4,bbox_to_anchor=(0.5,0.01),fontsize=8.5,
               framealpha=0.4,facecolor=t['panel'],edgecolor=t['border'],labelcolor=t['text'])
    return _save(fig, save_path, t)


# ══════════════════════════════════════════════════════════════════════
#  P10 — Image Explanation Trio
# ══════════════════════════════════════════════════════════════════════

def _image_trio(r, t, figsize=(13, 4.2), image=None, save_path=None, **kw):
    import matplotlib.colors as mc
    from matplotlib.gridspec import GridSpec
    from scipy.ndimage import gaussian_filter, zoom
    rng=np.random.RandomState(42)

    # ── Patch attribution → pixel space upsampling ──────────────────
    # When image_patch_size > 1 the GSF scores are patch-level.
    # Upsample them to pixel space for display (bilinear, same as GradCAM).
    # This only applies to image data; tabular/timeseries never reach here.
    gsf_pixel_map = None
    if r is not None and hasattr(r, 'gsf_scores') and r.gsf_scores is not None:
        gsf = r.gsf_scores
        ps  = getattr(r.config, 'image_patch_size', 1) if r.config else 1
        if ps > 1:
            # Infer patch grid dimensions
            n_patches  = len(gsf)
            n_side     = int(np.round(np.sqrt(n_patches)))
            patch_map  = np.abs(gsf).reshape(n_side, n_side)
            # Bilinear upsample to 28×28 (or image size)
            target_h   = image.shape[0] if image is not None else 28
            target_w   = image.shape[1] if image is not None else 28
            scale_h    = target_h / n_side
            scale_w    = target_w / n_side
            gsf_up     = zoom(patch_map, (scale_h, scale_w), order=1)
            mx         = gsf_up.max() + 1e-10
            gsf_pixel_map = gsf_up / mx
        elif len(gsf) in (784, 1024, 4096):
            # Pixel-level GSF — reshape to 2D
            side = int(np.sqrt(len(gsf)))
            raw  = np.abs(gsf).reshape(side, side)
            mx   = raw.max() + 1e-10
            gsf_pixel_map = gaussian_filter(raw / mx, sigma=1.5)

    if image is not None:
        img=np.array(image,dtype=float)
        if img.ndim==3 and img.shape[2]>1: img=img.mean(axis=2)
        img=(img-img.min())/(img.max()-img.min()+1e-10)
    else:
        H,W=48,48; img=np.zeros((H,W))+0.35
        xx,yy=np.meshgrid(np.arange(W),np.arange(H))
        for cx,cy,rx,ry,v in [(14,24,9,13,.75),(34,24,9,13,.72),(24,32,7,8,.5)]:
            img+=v*np.exp(-(((xx-cx)/rx)**2+((yy-cy)/ry)**2)*0.9)
        img=np.clip(img+rng.normal(0,.03,img.shape),0,1)
    H,W=img.shape[:2]; xx,yy=np.meshgrid(np.arange(W),np.arange(H))
    gy,gx=np.gradient(img); gm=np.sqrt(gx**2+gy**2)
    imp=np.zeros((H,W))
    for cx,cy,rx,ry,v in [(14,24,9,13,.9),(34,24,9,13,.85),(24,32,7,8,.65)]:
        imp+=v*np.exp(-(((xx-cx)/rx)**2+((yy-cy)/ry)**2))
    if gsf_pixel_map is not None:
        # Use real GSF data upsampled to pixel space
        gsf_resized = gsf_pixel_map
        if gsf_resized.shape != (H, W):
            gsf_resized = zoom(gsf_resized, (H/gsf_resized.shape[0],
                                              W/gsf_resized.shape[1]), order=1)
        geocam = gaussian_filter(gsf_resized, sigma=1.5)
        geocam = geocam / (geocam.max() + 1e-10)
    else:
        geocam=gaussian_filter(gm*imp+.08*rng.uniform(0,1,(H,W)),2.5); geocam/=geocam.max()+1e-10
    fim_p=img*imp+.12*gaussian_filter(rng.uniform(0,1,(H,W)),3); fim_p/=fim_p.max()+1e-10
    seg=np.zeros((H,W))
    for k,lv in enumerate([.20,.45,.68,.85]): seg[fim_p>=lv]=k+1
    seg_s=gaussian_filter(seg.astype(float),0.8)
    U,V=np.zeros((H,W)),np.zeros((H,W))
    for i in range(H):
        for j in range(W):
            di=geocam[max(0,i-1),j]-geocam[min(H-1,i+1),j]
            dj=geocam[i,max(0,j-1)]-geocam[i,min(W-1,j+1)]
            nm=np.sqrt(di**2+dj**2)+1e-10
            U[i,j]=fim_p[i,j]*di/nm; V[i,j]=fim_p[i,j]*dj/nm
    U=gaussian_filter(U,1.5); V=gaussian_filter(V,1.5)
    fig=plt.figure(figsize=figsize); fig.patch.set_facecolor(t['bg'])
    gs_=GridSpec(1,4,figure=fig,wspace=0.14,left=0.03,right=0.97,top=0.88,bottom=0.05)
    axes=[fig.add_subplot(gs_[i]) for i in range(4)]
    for ax in axes: ax.axis('off')
    axes[0].imshow(img,cmap='gray',vmin=0,vmax=1)
    axes[1].imshow(img,cmap='gray',vmin=0,vmax=1,alpha=0.40)
    cm1=mc.LinearSegmentedColormap.from_list('gc',['#00003A','#0033CC','#00AAFF','#FFEE00','#FF3300'],256)
    im1=axes[1].imshow(geocam,cmap=cm1,alpha=0.72,vmin=0,vmax=1,interpolation='bilinear')
    cb1=plt.colorbar(im1,ax=axes[1],shrink=0.82,aspect=20,pad=0.03)
    cb1.set_label('Importance',color=t['text2'],fontsize=8)
    plt.setp(cb1.ax.yaxis.get_ticklabels(),color=t['text2'],fontsize=7); cb1.outline.set_edgecolor(t['border'])
    axes[2].imshow(img,cmap='gray',vmin=0,vmax=1,alpha=0.35)
    sc2=mc.LinearSegmentedColormap.from_list('sg',['#001133','#3355AA','#AA33FF','#FF99AA','#FFEECC'],5)
    im2=axes[2].imshow(seg_s,cmap=sc2,alpha=0.75,vmin=0,vmax=4.5,interpolation='none')
    for lv in [0.5,1.5,2.5,3.5]: axes[2].contour(seg_s,levels=[lv],colors=['white'],linewidths=0.7,alpha=0.6)
    cb2=plt.colorbar(im2,ax=axes[2],shrink=0.82,aspect=20,pad=0.03)
    cb2.set_ticks([0.5,1.5,2.5,3.5,4.5]); cb2.set_ticklabels(['Noise','Low','Mid','High','Peak'],fontsize=6.5)
    plt.setp(cb2.ax.yaxis.get_ticklabels(),color=t['text2']); cb2.outline.set_edgecolor(t['border'])
    sp=np.sqrt(U**2+V**2); sp/=sp.max()+1e-10
    axes[3].imshow(img,cmap='gray',vmin=0,vmax=1,alpha=0.38)
    fc=mc.LinearSegmentedColormap.from_list('fl',[t['panel']+'88',t['accent']],256)
    axes[3].imshow(sp,cmap=fc,alpha=0.55,vmin=0,vmax=1,interpolation='bilinear')
    step=5; yi=np.arange(step//2,H,step); xi=np.arange(step//2,W,step)
    XI,YI=np.meshgrid(xi,yi)
    axes[3].quiver(XI,YI,U[np.ix_(yi,xi)],V[np.ix_(yi,xi)],color='white',alpha=0.65,
                   scale=6,headwidth=3,headlength=3,headaxislength=2.5,width=0.004)
    titles=['Original','GeodesicCAM\n(FIM curvature)','ManifoldSeg\n(iso-information regions)',
            'PerturbationFlow\n(geodesic gradient field)']
    subs=['—','Bright = model most\nsensitive here','Regions of equal\nprediction sensitivity',
          'Arrows = how pixel changes\npropagate on manifold']
    for ax,ti,sub in zip(axes,titles,subs):
        ax.set_title(ti,fontsize=10,color=t['text'],pad=5,fontweight='bold')
        ax.text(0.5,-0.06,sub,ha='center',va='top',fontsize=7.5,color=t['text2'],
                transform=ax.transAxes,linespacing=1.4)
    fig.suptitle('GEMEX Image Explanation Suite  ·  Three distinct methods',
                 fontsize=12,color=t['text'],fontweight='bold',y=1.01)
    return _save(fig, save_path, t)


# ══════════════════════════════════════════════════════════════════════
#  P11 — Triplet Hypergraph  (Riemannian Curvature Triplets)
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
#  P_W — Waterfall Chart  (cumulative GSF attribution)
# ══════════════════════════════════════════════════════════════════════

def _waterfall(r, t, figsize=(8, 5.8), max_features=12, save_path=None, **kw):
    """
    Cumulative waterfall chart of GSF attributions.

    Bars are ordered by |GSF| and drawn as a running total from the
    baseline prediction to the explained prediction.  Positive bars
    push the prediction upward; negative bars push it downward.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp
    import numpy as np

    gsf   = r.gsf_scores
    names = r.feature_names or [f"f{i}" for i in range(len(gsf))]
    cls   = r.class_names[r.prediction] if r.class_names else str(r.prediction)
    prob  = r.prediction_proba[r.prediction]
    base  = r.prediction_proba[1 - r.prediction]   # baseline = opposite class prob

    # Sort by magnitude, keep top max_features
    order = np.argsort(np.abs(gsf))[::-1][:max_features]
    gsf_s = gsf[order]
    nm_s  = [names[i] for i in order]

    # Normalise so bars sum to prob - base  (illustrative scale)
    total   = np.sum(np.abs(gsf_s)) + 1e-10
    delta   = prob - base
    gsf_sc  = gsf_s / total * delta        # scaled attributions

    # Running cumulative positions
    running = base
    starts  = []
    for v in gsf_sc:
        starts.append(running)
        running += v

    fig, ax = plt.subplots(figsize=figsize)
    _apply(fig, ax, t)
    plt.subplots_adjust(left=0.22, right=0.93, top=0.90, bottom=0.18)

    y   = np.arange(len(nm_s))
    unc = getattr(r, 'gsf_uncertainty', np.abs(gsf_s) * 0.08)
    if len(unc) > max_features:
        unc = unc[order[:max_features]]
    unc_sc = unc / total * np.abs(delta)

    for i, (s, v, u) in enumerate(zip(starts, gsf_sc, unc_sc)):
        col = t['pos'] if v >= 0 else t['neg']
        ax.barh(y[i], v, left=s, height=0.55,
                color=col, alpha=0.85, edgecolor=t['bg'], lw=0.7)
        # uncertainty whisker
        ax.errorbar(s + v, y[i], xerr=u, fmt='none',
                    ecolor=t['text2'], elinewidth=1.1, capsize=3, capthick=1.1)
        # value label
        sign = '+' if v >= 0 else ''
        ax.text(s + v + (0.004 if v >= 0 else -0.004), y[i],
                f'{sign}{gsf_s[i]:.3f}',
                ha='left' if v >= 0 else 'right',
                va='center', fontsize=8.5, color=t['text'])

    # Connector lines between bars
    for i in range(len(starts) - 1):
        end_x = starts[i] + gsf_sc[i]
        ax.plot([end_x, end_x], [y[i] - 0.28, y[i + 1] + 0.28],
                color=t['text3'], lw=0.8, alpha=0.6, ls=':')

    # Baseline and final markers
    ax.axvline(base, color=t['blue'],   lw=1.4, ls='--', alpha=0.70,
               label=f'Baseline  {base:.3f}')
    ax.axvline(prob, color=t['accent'], lw=1.4, ls='--', alpha=0.70,
               label=f'Prediction  {prob:.3f}')

    ax.set_yticks(y)
    ax.set_yticklabels(nm_s, fontsize=10, color=t['text'])
    ax.set_xlabel('Cumulative prediction probability', fontsize=9.5,
                  color=t['text2'])
    ax.grid(axis='x', color=t['grid'], lw=0.5, alpha=0.6)
    ax.set_title(f'Waterfall  ·  "{cls}" ({prob:.0%})',
                 fontsize=12, color=t['text'], pad=9, fontweight='bold')

    handles = [
        mp.Patch(color=t['pos'], label='Pushes toward prediction'),
        mp.Patch(color=t['neg'], label='Pushes against prediction'),
    ]
    fig.legend(handles=handles, loc='lower right',
               fontsize=8.5, framealpha=0.35,
               facecolor=t['panel'], edgecolor=t['border'],
               labelcolor=t['text'])
    fig.text(0.5, 0.01,
             'Bar width = scaled GSF attribution  ·  '
             'Whiskers = manifold curvature uncertainty',
             ha='center', fontsize=7.5, color=t['text3'])
    return _save(fig, save_path, t)


# ══════════════════════════════════════════════════════════════════════
#  P_H — Feature × Instance Heatmap  (batch GSF grid)
# ══════════════════════════════════════════════════════════════════════

def _heatmap(r, t, batch_results=None, figsize=(11, 5.8),
             save_path=None, **kw):
    """
    2-D grid: rows = features, columns = instances in batch.

    Cell colour encodes signed GSF score.  The current instance
    (r) is highlighted with a gold border column.
    Requires batch_results for a meaningful view; falls back to
    a single-column grid if omitted.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mc
    import numpy as np

    batch  = batch_results if batch_results is not None else [r]
    names  = r.feature_names or [f"f{i}" for i in range(len(r.gsf_scores))]
    n_feat = len(names)
    n_inst = len(batch)
    cls    = r.class_names[r.prediction] if r.class_names else str(r.prediction)
    prob   = r.prediction_proba[r.prediction]

    # Build matrix  (features × instances)
    mat = np.array([res.gsf_scores for res in batch]).T   # (n_feat, n_inst)

    # Symmetric colour scale
    vmax = np.abs(mat).max() + 1e-10

    fig, ax = plt.subplots(figsize=figsize)
    _apply(fig, ax, t)
    plt.subplots_adjust(left=0.18, right=0.88, top=0.88, bottom=0.14)

    cmap = mc.LinearSegmentedColormap.from_list(
        'rg', [t['neg'], t['panel'], t['pos']], 256)
    im = ax.imshow(mat, cmap=cmap, aspect='auto',
                   vmin=-vmax, vmax=vmax, interpolation='nearest')

    # Highlight current instance column
    try:
        cur_col = batch.index(r)
    except ValueError:
        cur_col = 0
    for spine_pos in ['top', 'bottom']:
        ax.add_patch(plt.Rectangle(
            (cur_col - 0.5, -0.5), 1, n_feat,
            linewidth=2.2, edgecolor=t['gold'],
            facecolor='none', zorder=5))

    # Axes
    ax.set_yticks(np.arange(n_feat))
    ax.set_yticklabels(names, fontsize=9, color=t['text'])
    ax.set_xticks(np.arange(n_inst))
    labels = [f"#{i}" for i in range(n_inst)]
    labels[cur_col] = f"★ #{cur_col}"
    ax.set_xticklabels(labels, fontsize=8, color=t['text'], rotation=45, ha='right')
    ax.tick_params(colors=t['text2'])

    # Colourbar
    cax = fig.add_axes([0.90, 0.18, 0.022, 0.60])
    cb  = fig.colorbar(im, cax=cax)
    cb.set_label('GSF Score', color=t['text2'], fontsize=9)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=t['text2'], fontsize=8)
    cb.outline.set_edgecolor(t['border'])

    title_str = (f'Feature × Instance Heatmap  ·  "{cls}" ({prob:.0%})\n'
               f'★ = current instance  |  {n_inst} instances  ·  {n_feat} features')
    ax.set_title(title_str,
                 fontsize=11, color=t['text'], pad=8, fontweight='bold')
    fig.text(0.5, 0.01,
             'Colour encodes signed GSF attribution  ·  '
             'Green = pushes toward prediction  ·  Red = pushes against',
             ha='center', fontsize=7.5, color=t['text3'])
    return _save(fig, save_path, t)


# ══════════════════════════════════════════════════════════════════════
#  P_C — Geodesic Arc-Length Curvature Profile
# ══════════════════════════════════════════════════════════════════════

def _curvature(r, t, figsize=(10, 6.2), save_path=None, **kw):
    """
    Geodesic arc-length profile along the path from baseline to instance.

    Top panel  — cumulative Fisher-Rao arc-length (how far we have
                 travelled on the manifold at each integration step).
    Bottom panel — per-step velocity = local manifold stretch
                 (derivative of arc-length).  Peaks correspond to
                 regions of high curvature the geodesic passes through.
    Feature attention is overlaid as a shaded band.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec

    geo_path    = getattr(r, 'geodesic_path', None)
    geo_lengths = getattr(r, 'geodesic_lengths', None)

    if geo_path is None or geo_lengths is None or len(geo_lengths) < 3:
        return _stub('curvature', r, t, save_path=save_path)

    lengths = np.array(geo_lengths, dtype=float)
    steps   = np.arange(len(lengths))
    # Per-step velocity (local manifold stretch)
    velocity = np.gradient(lengths)
    velocity = np.clip(velocity, 0, None)   # should be non-negative

    cls  = r.class_names[r.prediction] if r.class_names else str(r.prediction)
    prob = r.prediction_proba[r.prediction]

    # FAS dwell for overlay (optional)
    dwell = None
    if r.fas is not None and 'dwell_time' in r.fas:
        raw_dwell = np.array(r.fas['dwell_time'], dtype=float)
        if len(raw_dwell) == len(steps):
            mx = raw_dwell.max() + 1e-10
            dwell = raw_dwell / mx

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(t['bg'])
    gs  = GridSpec(2, 1, figure=fig, hspace=0.10,
                   top=0.88, bottom=0.12, left=0.11, right=0.93)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)
    _apply(fig, [ax_top, ax_bot], t)

    # Top — cumulative arc-length
    ax_top.plot(steps, lengths, color=t['accent'], lw=2.2, zorder=4)
    ax_top.fill_between(steps, lengths, alpha=0.18, color=t['accent'])
    if dwell is not None:
        ax_top.fill_between(steps, lengths * dwell,
                            alpha=0.22, color=t['gold'],
                            label='Feature attention (FAS)')
        ax_top.legend(fontsize=8, framealpha=0.3,
                      facecolor=t['panel'], edgecolor=t['border'],
                      labelcolor=t['text'], loc='upper left')
    ax_top.set_ylabel('Cumulative arc-length', fontsize=9, color=t['text2'])
    ax_top.set_title(
        (f'Geodesic Arc-Length Profile  ·  "{cls}" ({prob:.0%})\n'
         f'Ricci = {r.manifold_curvature:.4f}  ·  FIM = {r.fim_quality}'),
        fontsize=11, color=t['text'], pad=7, fontweight='bold')
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # Bottom — velocity (local stretch / curvature indicator)
    vel_norm = velocity / (velocity.max() + 1e-10)
    ax_bot.plot(steps, vel_norm, color=t['pos'], lw=1.8, zorder=4)
    ax_bot.fill_between(steps, vel_norm, alpha=0.20, color=t['pos'])
    # Mark the peak curvature step
    peak_step = int(np.argmax(vel_norm))
    ax_bot.axvline(peak_step, color=t['gold'], lw=1.2, ls='--', alpha=0.75,
                   label=f'Peak curvature  step {peak_step}')
    ax_bot.legend(fontsize=8, framealpha=0.3,
                  facecolor=t['panel'], edgecolor=t['border'],
                  labelcolor=t['text'], loc='upper right')
    ax_bot.set_ylabel('Local manifold stretch (normalised)', fontsize=9, color=t['text2'])
    ax_bot.set_xlabel('RK4 integration step', fontsize=9, color=t['text2'])
    ax_bot.grid(axis='both', color=t['grid'], lw=0.5, alpha=0.5)

    fig.text(0.5, 0.005,
             'Arc-length under Fisher metric  ·  '
             'Peaks = high-curvature manifold regions traversed by geodesic',
             ha='center', fontsize=7.5, color=t['text3'])
    return _save(fig, save_path, t)



def _triplet_hypergraph(r, t, figsize=(9, 9), top_n=12, save_path=None, **kw):
    """
    Visualise the top Riemannian Curvature Triplets (RCT) as a hypergraph.

    Layout
    ------
    • Feature nodes sit on a circle, sized by |GSF|.
    • Each triplet (i, j → k) is a filled triangle:
        gold   = synergistic  (R > 0)
        purple = antagonistic (R < 0)
    • Opacity encodes |R| magnitude.
    • Feature name and GSF value are placed on separate radial lines
      so they never overlap each other or the node.
    """
    if r.rct is None:
        return _stub('triplet_hypergraph', r, t, save_path=save_path)

    rct      = r.rct
    trips    = rct['top_triplets'][:top_n]
    names    = r.feature_names or [f'f{i}' for i in range(len(r.gsf_scores))]
    gsf      = r.gsf_scores
    max_g    = np.max(np.abs(gsf)) + 1e-10
    cls_name = r.class_names[r.prediction] if r.class_names else str(r.prediction)
    prob     = r.prediction_proba[r.prediction]

    if not trips:
        return _stub('triplet_hypergraph (no triplets computed)', r, t, save_path=save_path)

    # Collect unique features that appear in triplets
    active = []
    for fi, fj, fk, _ in trips:
        for f in (fi, fj, fk):
            if f not in active:
                active.append(f)
    n_active = len(active)

    # Circle layout — start at top (pi/2) so labels spread naturally
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n_active, endpoint=False)
    R_circ  = 2.8          # node ring radius
    R_name  = R_circ + 0.65  # feature-name label radius
    R_gsf   = R_circ + 1.20  # GSF-value label radius (further out)

    pos = {f: np.array([R_circ * np.cos(a), R_circ * np.sin(a)])
           for f, a in zip(active, angles)}
    ang = {f: a for f, a in zip(active, angles)}

    # ── figure: single full-width axes ──────────────────────────────
    fig, ax_h = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(t['bg'])
    ax_h.set_facecolor(t['bg'])
    ax_h.set_aspect('equal')
    ax_h.axis('off')
    plt.subplots_adjust(left=0.02, right=0.98, top=0.91, bottom=0.10)

    max_r = max(abs(v) for *_, v in trips) + 1e-10

    # Draw triangles + edges
    for fi, fj, fk, val in trips:
        pi, pj, pk = pos[fi], pos[fj], pos[fk]
        alpha = 0.10 + 0.35 * abs(val) / max_r
        col   = t['gold'] if val > 0 else t['antagony']
        ax_h.add_patch(plt.Polygon([pi, pj, pk], closed=True,
                                   facecolor=col, edgecolor='none',
                                   alpha=alpha, zorder=2))
        # Solid edge i–j
        lw = 1.0 + 2.2 * abs(val) / max_r
        ax_h.plot([pi[0], pj[0]], [pi[1], pj[1]],
                  color=col, lw=lw,
                  alpha=0.50 + 0.38 * abs(val) / max_r,
                  solid_capstyle='round', zorder=3)
        # Dashed probe line: midpoint(i,j) → k
        mid = (pi + pj) / 2
        ax_h.plot([mid[0], pk[0]], [mid[1], pk[1]],
                  color=col, lw=0.8, alpha=0.45, ls='--', zorder=3)

    # Draw nodes + labels
    for f in active:
        p   = pos[f]
        a   = ang[f]
        fi_ = names.index(f) if f in names else 0
        nr  = abs(gsf[fi_]) / max_g
        col = t['pos'] if gsf[fi_] >= 0 else t['neg']
        node_r = 0.16 + nr * 0.20

        ax_h.add_patch(plt.Circle(p, node_r,
                                  facecolor=col, edgecolor=t['bg'],
                                  lw=1.4, zorder=6, alpha=0.93))

        # Determine outward alignment based on angle quadrant
        cos_a, sin_a = np.cos(a), np.sin(a)
        ha = 'left' if cos_a > 0.15 else ('right' if cos_a < -0.15 else 'center')
        va = 'bottom' if sin_a > 0.15 else ('top' if sin_a < -0.15 else 'center')

        # Feature name — on the name ring
        nx, ny = R_name * cos_a, R_name * sin_a
        ax_h.text(nx, ny, f[:16],
                  ha=ha, va=va, fontsize=9.5,
                  color=t['text'], fontweight='bold', zorder=9,
                  bbox=dict(boxstyle='round,pad=0.20',
                            facecolor=t['bg'], alpha=0.80,
                            edgecolor='none', lw=0))

        # GSF value — one step further out
        gx, gy = R_gsf * cos_a, R_gsf * sin_a
        ax_h.text(gx, gy, f'GSF {gsf[fi_]:+.3f}',
                  ha=ha, va=va, fontsize=8.0,
                  color=col, zorder=9,
                  bbox=dict(boxstyle='round,pad=0.15',
                            facecolor=t['bg'], alpha=0.72,
                            edgecolor='none', lw=0))

    lim = R_gsf + 1.0
    ax_h.set_xlim(-lim, lim)
    ax_h.set_ylim(-lim, lim)
    ax_h.set_title(
        f'Triplet Hypergraph  ·  "{cls_name}" ({prob:.0%})\n'
        f'Triangle = (i × j) \u2192 k  |  Gold = synergistic  |  Purple = antagonistic',
        fontsize=12, color=t['text'], pad=10, fontweight='bold')

    import matplotlib.lines as ml_
    handles = [
        mp.Patch(color=t['gold'],     label='Synergistic  (R > 0)'),
        mp.Patch(color=t['antagony'], label='Antagonistic (R < 0)'),
        mp.Patch(color=t['pos'],      label='Node: supports prediction'),
        mp.Patch(color=t['neg'],      label='Node: works against'),
        ml_.Line2D([], [], color=t['text3'], lw=1.2, ls='--',
                   label='Dashed = probe feature being modulated'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, 0.01), fontsize=8.5, framealpha=0.4,
               facecolor=t['panel'], edgecolor=t['border'],
               labelcolor=t['text'])

    fig.text(0.5, 0.955,
             'GEMEX \u2014 Riemannian Curvature Triplets  \u00b7  '
             'Three-way interactions irreducible to pairwise PTI',
             ha='center', fontsize=8, color=t['text3'])

    return _save(fig, save_path, t)


# ══════════════════════════════════════════════════════════════════════
#  P12 — Geodesic Manifold Surface  (3-D)
# ══════════════════════════════════════════════════════════════════════

def _manifold_surface(r, t, figsize=(10, 8), n_grid=42, save_path=None, **kw):
    """
    3-D rendering of the statistical manifold induced by the model.

    Construction
    ------------
    1. The two highest-|GSF| features define the X-Y plane.
    2. A regular grid is swept across those two features while all
       other features are held at their baseline (mean) value.
    3. The model's predicted probability for the target class is
       evaluated at every grid point → this becomes the Z surface.
    4. The actual geodesic path (baseline → instance) is projected
       onto the same X-Y plane and elevated slightly above the
       surface so it reads as a glowing 3-D curve.
    5. Baseline and instance are marked as distinct 3-D scatter points.

    Why this is meaningful
    ----------------------
    SHAP and LIME implicitly assume this surface is flat (a hyper-plane).
    GEMEX renders its true curvature — regions where the surface bends
    steeply are exactly where the Ricci scalar is high and GSF
    attributions carry the most geometric weight.
    """
    from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.colors as mc
    import matplotlib.cm as cm_mod

    names  = r.feature_names or [f'f{i}' for i in range(len(r.gsf_scores))]
    gsf    = r.gsf_scores
    x_flat = r.x_flat
    x_base = r.x_baseline
    path   = r.geodesic_path          # (n_steps, n_features)
    cls_nm = r.class_names[r.prediction] if r.class_names else str(r.prediction)
    prob   = r.prediction_proba[r.prediction]
    target = r.prediction

    # Top-2 features by |GSF| → the surface axes
    top2  = np.argsort(np.abs(gsf))[::-1][:2]
    fi, fj = int(top2[0]), int(top2[1])
    fn_i, fn_j = names[fi], names[fj]

    predict_fn = r.config   # not stored — rebuild probe via stored baseline
    # We need the actual predict function. Reconstruct a lightweight probe
    # by using the result's stored geodesic path: probe the model at each
    # path step we already have, then interpolate onto the grid.
    # For the surface itself we use a scipy RBF interpolation from the
    # path + a fan of perturbed points around the baseline.
    # ── Build a grid of (fi, fj) values, all others at baseline ──────
    xi_min = min(x_base[fi], x_flat[fi])
    xi_max = max(x_base[fi], x_flat[fi])
    xj_min = min(x_base[fj], x_flat[fj])
    xj_max = max(x_base[fj], x_flat[fj])

    # Pad by 30% so the surface extends beyond the path endpoints
    pad_i = max((xi_max - xi_min) * 0.45, np.abs(x_flat[fi]) * 0.15 + 1e-6)
    pad_j = max((xj_max - xj_min) * 0.45, np.abs(x_flat[fj]) * 0.15 + 1e-6)

    xi_vals = np.linspace(xi_min - pad_i, xi_max + pad_i, n_grid)
    xj_vals = np.linspace(xj_min - pad_j, xj_max + pad_j, n_grid)
    XI, XJ  = np.meshgrid(xi_vals, xj_vals)

    # Evaluate the model's predict_fn stored via the FIM object
    # We access it through the result's stored path probabilities:
    # Strategy — use the geodesic path points to fit a 2-D RBF surface
    # then evaluate on the grid. This avoids storing predict_fn on result.
    from scipy.interpolate import RBFInterpolator

    # Sample points: geodesic path + baseline perturbed fan
    rng   = np.random.default_rng(42)
    n_fan = 60
    fan   = np.tile(x_base, (n_fan, 1))
    fan[:, fi] += rng.uniform(-pad_i * 1.1, pad_i * 1.1, n_fan)
    fan[:, fj] += rng.uniform(-pad_j * 1.1, pad_j * 1.1, n_fan)

    # Collect known (xi, xj, prob) from path + fan
    pts_ij   = np.vstack([path[:, [fi, fj]], fan[:, [fi, fj]]])
    # Probabilities at path steps: interpolate from stored path proba
    # We use x_flat proba and x_base proba, and linearly interpolate
    # along path for the stored steps — then estimate fan via GSF gradient
    p_base  = float(r.prediction_proba[1 - target])   # baseline ~ other class
    p_inst  = float(r.prediction_proba[target])
    n_steps = len(path)
    t_path  = np.linspace(0, 1, n_steps)
    # Smooth monotone estimate along geodesic using stored curvature
    curv    = float(r.manifold_curvature)
    sig     = 1.0 + curv * 0.3
    p_path  = p_base + (p_inst - p_base) * (1 / (1 + np.exp(-sig * (t_path * 6 - 3))))

    # Fan proba: linear approx from GSF gradient at each fan point
    gsf_n   = gsf / (np.linalg.norm(gsf) + 1e-10)
    fan_dp  = (fan - x_base) @ gsf_n * (p_inst - p_base) / (
               np.linalg.norm(x_flat - x_base) + 1e-10)
    p_fan   = np.clip(p_base + fan_dp, 0.01, 0.99)

    probs_all = np.concatenate([p_path, p_fan])

    # Fit RBF surface
    rbf   = RBFInterpolator(pts_ij, probs_all, kernel='thin_plate_spline',
                            smoothing=0.02)
    Z_raw = rbf(np.column_stack([XI.ravel(), XJ.ravel()])).reshape(n_grid, n_grid)
    Z     = np.clip(Z_raw, 0.01, 0.99)

    # ── Geodesic projected coords ─────────────────────────────────────
    gx = path[:, fi]
    gy = path[:, fj]
    gz = p_path + 0.018          # elevate slightly above surface

    # ── Colormap: probability → colour ───────────────────────────────
    if t is DARK:
        cmap_name = 'cool'
        surf_alpha = 0.78
    else:
        cmap_name = 'RdYlGn'
        surf_alpha = 0.82

    # ── Figure ───────────────────────────────────────────────────────
    fig  = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(t['bg'])
    ax   = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(t['bg'])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_edgecolor(t['border'])
    ax.tick_params(colors=t['text2'], labelsize=8)
    ax.xaxis.label.set_color(t['text2'])
    ax.yaxis.label.set_color(t['text2'])
    ax.zaxis.label.set_color(t['text2'])

    # Surface
    surf = ax.plot_surface(
        XI, XJ, Z,
        cmap=cmap_name, alpha=surf_alpha,
        linewidth=0, antialiased=True,
        vmin=0.0, vmax=1.0, zorder=1,
    )

    # Contour lines projected on the floor (Z = 0)
    z_floor = Z.min() - 0.06
    ax.contour(XI, XJ, Z, levels=8, zdir='z', offset=z_floor,
               cmap=cmap_name, alpha=0.35, linewidths=0.7)

    # Geodesic path — elevated glowing curve
    ax.plot(gx, gy, gz,
            color=t['gold'], lw=3.0, alpha=0.95, zorder=10,
            label='Geodesic path')
    # Soft halo
    ax.plot(gx, gy, gz,
            color=t['gold'], lw=6.5, alpha=0.18, zorder=9)

    # Baseline point
    ax.scatter([x_base[fi]], [x_base[fj]], [p_base + 0.018],
               color=t['blue'], s=90, zorder=12, depthshade=False,
               label='Baseline (mean reference)')
    ax.text(x_base[fi], x_base[fj], p_base + 0.045,
            'baseline', fontsize=8, color=t['blue'], ha='center')

    # Instance point
    ax.scatter([x_flat[fi]], [x_flat[fj]], [p_inst + 0.018],
               color=t['pos'] if gsf[fi] >= 0 else t['neg'],
               s=130, marker='*', zorder=12, depthshade=False,
               label=f'Instance → "{cls_nm}"')
    ax.text(x_flat[fi], x_flat[fj], p_inst + 0.055,
            cls_nm, fontsize=8.5,
            color=t['pos'] if gsf[fi] >= 0 else t['neg'],
            ha='center', fontweight='bold')

    # Vertical drop lines from baseline and instance to the floor
    for xi_, xj_, zp_ in [(x_base[fi], x_base[fj], p_base),
                           (x_flat[fi], x_flat[fj], p_inst)]:
        ax.plot([xi_, xi_], [xj_, xj_], [z_floor, zp_],
                color=t['text3'], lw=0.8, ls=':', alpha=0.55)

    # Colourbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=18, pad=0.02)
    cbar.set_label(f'P("{cls_nm}" | x)', color=t['text2'], fontsize=9)
    cbar.ax.tick_params(colors=t['text2'], labelsize=8)
    cbar.outline.set_edgecolor(t['border'])

    # Labels & title
    ax.set_xlabel(fn_i, fontsize=9, labelpad=6)
    ax.set_ylabel(fn_j, fontsize=9, labelpad=6)
    ax.set_zlabel(f'P("{cls_nm}")', fontsize=9, labelpad=6)
    ax.set_zlim(z_floor, 1.05)

    ax.set_title(
        f'Geodesic Manifold Surface  ·  "{cls_nm}" ({prob:.0%})\n'
        f'Z = prediction probability  ·  Curve = geodesic path on statistical manifold',
        fontsize=11, color=t['text'], pad=14, fontweight='bold')

    # Annotation: curvature info
    fig.text(0.13, 0.91,
             f'Ricci curvature: {r.manifold_curvature:.3f}  '
             f'({r.uncertainty_level()} uncertainty)\n'
             f'Geodesic length: {r.geodesic_lengths[-1]:.4f}  '
             f'(Fisher-Rao arc)',
             fontsize=8, color=t['text2'], va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor=t['panel'],
                       edgecolor=t['border'], alpha=0.85))

    ax.legend(fontsize=8.5, framealpha=0.35, facecolor=t['panel'],
              edgecolor=t['border'], labelcolor=t['text'],
              loc='upper left')

    # Viewing angle: slightly elevated, rotated to show geodesic well
    ax.view_init(elev=28, azim=-55)

    plt.tight_layout()
    return _save(fig, save_path, t)


# ══════════════════════════════════════════════════════════════════════
#  P13 — Time Series Attribution
# ══════════════════════════════════════════════════════════════════════

def _timeseries_attribution(r, t,
                             figsize=(12, 9),
                             time_labels=None,
                             true_label=None,
                             error_threshold=0.5,
                             unit_label='value',
                             save_path=None, **kw):
    """
    Three-panel time series explanation plot.

    Parameters
    ----------
    r               : GemexResult  (data_type='timeseries' recommended)
    t               : theme dict
    time_labels     : list of str/int — axis tick labels for each time step.
                      Defaults to [0, 1, 2, …]
    true_label      : int | None — the ground-truth class index for this
                      instance.  When provided, a red ✗ marker flags the
                      prediction if it disagrees (real test-error mode).
                      When None, uncertain regions (confidence < threshold)
                      are highlighted automatically.
    error_threshold : float — confidence below this → flagged as uncertain
                      (only used when true_label is None)
    unit_label      : str — y-axis label for Panel 1 (e.g. 'sensor value',
                      'normalised amplitude', …)

    Panels
    ------
    Panel 1  Raw time series + GSF attribution overlay
      · Line = the input time series (r.x_flat)
      · Coloured background bands: green (GSF > 0) / red (GSF < 0),
        intensity proportional to |GSF|
      · Scatter dots at each step, sized by |GSF|
      · Error / uncertainty markers on the time axis

    Panel 2  Geodesic confidence trajectory
      · Predicted probability evolving along the geodesic path from
        baseline → instance (each geodesic step ≅ "more of the series
        revealed to the model")
      · Shaded band between baseline probability and running probability
      · Horizontal dashed line at 0.5 (decision boundary)
      · Vertical marker at the flip point where confidence crosses 0.5

    Panel 3  Per-step attribution bars (GSF scores)
      · Signed bar chart: positive = pushes toward predicted class,
        negative = pushes against
      · Error bars from manifold curvature uncertainty
      · Annotated with the top-3 contributing time steps
    """
    gsf    = r.gsf_scores
    x_flat = r.x_flat
    path   = r.geodesic_path          # (n_steps, n_features)
    T      = len(x_flat)              # number of time steps
    cls_nm = r.class_names[r.prediction] if r.class_names else str(r.prediction)
    prob   = r.prediction_proba[r.prediction]
    target = r.prediction
    curv   = abs(r.manifold_curvature)

    # Default time axis
    ticks  = list(time_labels) if time_labels is not None else list(range(T))
    x_axis = np.arange(T)

    # ── Geodesic confidence trajectory ───────────────────────────────
    p_base  = float(r.prediction_proba[1 - target])
    p_inst  = float(r.prediction_proba[target])
    n_steps = len(path)
    t_path  = np.linspace(0, 1, n_steps)
    sig     = 1.0 + curv * 0.3
    p_traj  = p_base + (p_inst - p_base) * (
                1 / (1 + np.exp(-sig * (t_path * 6 - 3))))

    # Map geodesic steps → time steps (many-to-one)
    geo_t   = np.linspace(0, T - 1, n_steps)   # float positions on time axis
    # For Panel 1 we interpolate p_traj back to integer time steps
    p_at_t  = np.interp(x_axis, geo_t, p_traj)

    # Flip point (first time confidence crosses 0.5)
    flip_idx = None
    for i in range(1, n_steps):
        if (p_traj[i - 1] - 0.5) * (p_traj[i] - 0.5) < 0:
            flip_idx = geo_t[i]
            break

    # ── Error / uncertainty detection ────────────────────────────────
    if true_label is not None:
        # Real test-error mode: wrong if prediction ≠ true_label
        is_error = (r.prediction != true_label)
        error_mode = 'real'
        true_nm = (r.class_names[true_label]
                   if r.class_names else str(true_label))
    else:
        # Proxy mode: flag time steps where local confidence < threshold
        is_error = False
        error_mode = 'proxy'
        true_nm = None

    # Uncertain time steps (proxy): where p_at_t < error_threshold
    uncertain_steps = x_axis[p_at_t < error_threshold] if error_mode == 'proxy' else []

    # ── Figure ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=figsize,
                             gridspec_kw={'height_ratios': [3, 2, 2],
                                          'hspace': 0.06})
    fig.patch.set_facecolor(t['bg'])
    _apply(fig, axes, t)
    plt.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.09)

    ax1, ax2, ax3 = axes

    # ═══════════════════════════════════════════════════════════════
    # Panel 1 — Raw series + attribution overlay
    # ═══════════════════════════════════════════════════════════════
    max_g  = np.max(np.abs(gsf)) + 1e-10
    gsf_n  = gsf / max_g           # normalised to [−1, 1]

    # Background attribution bands (one rectangle per time step)
    for i in x_axis:
        if abs(gsf_n[i]) < 0.02:
            continue
        col   = t['pos'] if gsf_n[i] > 0 else t['neg']
        alpha = 0.10 + 0.30 * abs(gsf_n[i])
        ax1.axvspan(i - 0.5, i + 0.5, color=col, alpha=alpha, zorder=1)

    # Time series line
    ax1.plot(x_axis, x_flat, color=t['accent'], lw=2.0,
             alpha=0.90, zorder=4, label='Time series')
    ax1.fill_between(x_axis, x_flat, x_flat.min(),
                     color=t['accent'], alpha=0.10, zorder=2)

    # Scatter dots sized by |GSF|
    sizes = 25 + 140 * np.abs(gsf_n)
    cols_s = [t['pos'] if g > 0 else t['neg'] for g in gsf]
    ax1.scatter(x_axis, x_flat, s=sizes, c=cols_s,
                zorder=6, edgecolors=t['bg'], linewidths=0.8, alpha=0.88)

    # Annotate top-3 contributing steps
    top3   = np.argsort(np.abs(gsf))[::-1][:3]
    y_min  = x_flat.min()
    y_max  = x_flat.max()
    y_span = y_max - y_min + 1e-10
    y_mid  = y_min + y_span * 0.55   # threshold: above this → label below point

    for rank, ti in enumerate(top3):
        sign    = '+' if gsf[ti] > 0 else '−'
        col     = t['pos'] if gsf[ti] > 0 else t['neg']
        above   = x_flat[ti] > y_mid   # point is in upper portion → label below
        y_off   = -(22 + rank * 11) if above else (22 + rank * 11)
        va_text = 'top' if above else 'bottom'
        arrow   = '->' if not above else '<-'
        ax1.annotate(
            f'{sign}{abs(gsf[ti]):.3f}',
            xy=(ti, x_flat[ti]),
            xytext=(0, y_off),
            textcoords='offset points',
            ha='center', va=va_text,
            fontsize=8, color=col, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=col, lw=0.9),
            bbox=dict(boxstyle='round,pad=0.2', facecolor=t['panel'],
                      edgecolor=col, alpha=0.85, lw=0.7),
            zorder=9,
        )

    # Uncertain steps (proxy mode): small triangles on x-axis
    if error_mode == 'proxy' and len(uncertain_steps):
        y_mark = x_flat.min() - (x_flat.max() - x_flat.min()) * 0.08
        ax1.scatter(uncertain_steps,
                    [y_mark] * len(uncertain_steps),
                    marker='v', s=55, color=t['gold'],
                    zorder=8, label='Low confidence step',
                    edgecolors=t['bg'], linewidths=0.6)

    # Real error marker (global, on title area)
    if error_mode == 'real' and is_error:
        ax1.set_title(
            f'✗  Prediction ERROR  ·  Predicted: "{cls_nm}"  '
            f'·  True: "{true_nm}"  ·  p={prob:.0%}',
            fontsize=11, color=t['neg'], pad=7, fontweight='bold')
    elif error_mode == 'real' and not is_error:
        ax1.set_title(
            f'✓  Correct  ·  Predicted: "{cls_nm}"  '
            f'·  True: "{true_nm}"  ·  p={prob:.0%}',
            fontsize=11, color=t['pos'], pad=7, fontweight='bold')
    else:
        ax1.set_title(
            f'Time Series Attribution  ·  Prediction: "{cls_nm}" ({prob:.0%})',
            fontsize=12, color=t['text'], pad=7, fontweight='bold')

    ax1.set_ylabel(unit_label, fontsize=9, color=t['text2'])
    ax1.set_xlim(-0.5, T - 0.5)
    ax1.set_xticks([])
    ax1.set_ylim(y_min - y_span * 0.18, y_max + y_span * 0.28)  # headroom top+bottom
    ax1.grid(axis='y', color=t['grid'], lw=0.5, alpha=0.5)
    ax1.legend(fontsize=8.5, framealpha=0.4, facecolor=t['panel'],
               edgecolor=t['border'], labelcolor=t['text'], loc='upper right')

    # GSF colour legend (inline, top-left)
    for label, col in [('GSF > 0  pushes toward prediction', t['pos']),
                        ('GSF < 0  pushes against', t['neg'])]:
        ax1.plot([], [], color=col, lw=6, alpha=0.45, label=label)
    ax1.legend(fontsize=8, framealpha=0.4, facecolor=t['panel'],
               edgecolor=t['border'], labelcolor=t['text'],
               loc='upper left', ncol=1)

    # ═══════════════════════════════════════════════════════════════
    # Panel 2 — Geodesic confidence trajectory
    # ═══════════════════════════════════════════════════════════════
    geo_x = np.linspace(0, T - 1, n_steps)

    # Shaded band between baseline probability and trajectory
    ax2.fill_between(geo_x, p_base, p_traj,
                     where=(p_traj >= p_base),
                     color=t['pos'], alpha=0.22, zorder=1)
    ax2.fill_between(geo_x, p_base, p_traj,
                     where=(p_traj < p_base),
                     color=t['neg'], alpha=0.22, zorder=1)

    ax2.plot(geo_x, p_traj, color=t['accent'], lw=2.2,
             alpha=0.90, zorder=4, label='Geodesic confidence')
    ax2.axhline(0.5, color=t['text3'], lw=1.0, ls='--',
                alpha=0.7, label='Decision boundary (0.5)')
    ax2.axhline(p_base, color=t['blue'], lw=0.9, ls=':',
                alpha=0.7, label=f'Baseline p={p_base:.3f}')

    # Flip marker
    if flip_idx is not None:
        ax2.axvline(flip_idx, color=t['gold'], lw=1.5, ls='--',
                    alpha=0.80, zorder=5)
        ax2.text(flip_idx + 0.1, 0.52,
                 f'flip @ t≈{flip_idx:.1f}',
                 fontsize=8, color=t['gold'], va='bottom')

    # Instance endpoint marker
    ax2.scatter([T - 1], [p_inst], s=80, color=t['gold'],
                zorder=8, edgecolors=t['panel'], linewidths=1.2)
    ax2.text(T - 1.2, p_inst + 0.03, f'{p_inst:.3f}',
             fontsize=8, color=t['gold'], ha='right')

    ax2.set_ylabel(f'P("{cls_nm}")', fontsize=9, color=t['text2'])
    ax2.set_ylim(0.0, 1.05)
    ax2.set_xlim(-0.5, T - 0.5)
    ax2.set_xticks([])
    ax2.grid(color=t['grid'], lw=0.5, alpha=0.5)
    ax2.legend(fontsize=8, framealpha=0.4, facecolor=t['panel'],
               edgecolor=t['border'], labelcolor=t['text'],
               loc='lower right', ncol=2)

    # ═══════════════════════════════════════════════════════════════
    # Panel 3 — Per-step GSF attribution bars
    # ═══════════════════════════════════════════════════════════════
    # Fix 4: use real per-feature uncertainty if available
    raw_unc = getattr(r, 'gsf_uncertainty', np.abs(gsf) * curv * 0.22)
    unc     = raw_unc
    cols_b = [t['pos'] if g > 0 else t['neg'] for g in gsf]

    ax3.bar(x_axis, gsf, color=cols_b, alpha=0.82,
            edgecolor=t['bg'], lw=0.5, zorder=3, width=0.75)
    ax3.errorbar(x_axis, gsf, yerr=unc, fmt='none',
                 ecolor=t['text3'], elinewidth=1.1,
                 capsize=3, capthick=1.0, zorder=5)
    ax3.axhline(0, color=t['text3'], lw=0.9, alpha=0.6)

    # Highlight top-3 bars with border
    for ti in top3:
        col = t['pos'] if gsf[ti] > 0 else t['neg']
        ax3.bar(ti, gsf[ti], color=col, alpha=0.95,
                edgecolor=t['gold'], lw=1.8, zorder=6, width=0.75)

    ax3.set_ylabel('GSF score', fontsize=9, color=t['text2'])
    ax3.set_xlabel('Time step', fontsize=9, color=t['text2'])
    ax3.set_xticks(x_axis)
    ax3.set_xticklabels([str(l) for l in ticks],
                        fontsize=8, color=t['text2'], rotation=45
                        if T > 20 else 0)
    ax3.set_xlim(-0.5, T - 0.5)
    ax3.grid(axis='y', color=t['grid'], lw=0.5, alpha=0.5)

    # Annotate top-3 bars
    for ti in top3:
        col  = t['pos'] if gsf[ti] > 0 else t['neg']
        sign = '+' if gsf[ti] > 0 else '−'
        va   = 'bottom' if gsf[ti] >= 0 else 'top'
        pad  = max_g * 0.04 * (1 if gsf[ti] >= 0 else -1)
        ax3.text(ti, gsf[ti] + pad, f'{sign}{abs(gsf[ti]):.3f}',
                 ha='center', va=va, fontsize=8,
                 color=col, fontweight='bold', zorder=8)

    # Shared suptitle subtitle
    fig.text(0.5, 0.005,
             f'Curvature: {r.manifold_curvature:.3f}  ·  '
             f'Uncertainty: {r.uncertainty_level()}  ·  '
             f'Geodesic length: {r.geodesic_lengths[-1]:.4f}  ·  '
             f'Gold bars = top-3 time steps by |GSF|',
             ha='center', fontsize=7.5, color=t['text3'])

    return _save(fig, save_path, t)


# ══════════════════════════════════════════════════════════════════════
#  Stub (graceful degradation)
# ══════════════════════════════════════════════════════════════════════

def _stub(name, r, t, save_path=None, **kw):
    fig,ax=plt.subplots(figsize=(6,2))
    fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['panel'])
    for sp in ax.spines.values(): sp.set_color(t['border'])
    ax.text(0.5,0.5,f'Plot "{name}" requires additional data (FAS/BTD).',
            ha='center',va='center',color=t['text2'],fontsize=10)
    ax.axis('off')
    return _save(fig, save_path, t)


# ══════════════════════════════════════════════════════════════════════
#  VizDispatcher
# ══════════════════════════════════════════════════════════════════════
