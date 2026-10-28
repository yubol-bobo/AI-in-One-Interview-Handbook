"""Regenerate handbook attention schematics: python scripts/generate_attention_diagrams.py.

Requires matplotlib. Original pedagogical diagrams, not benchmark results.
SVG keeps editable text; PNG is the GitHub inline preview.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figs'
C = dict(ink='#172b4d', muted='#52657a', line='#62768b', blue='#2669a8',
         pale='#edf5fc', purple='#7953a6', cache='#f0eafb', green='#21856d',
         mint='#e9f5ef', amber='#a76a0b', sand='#fff3d9', border='#cbd6e2')
plt.rcParams.update({'font.family':'DejaVu Sans', 'svg.fonttype':'none',
                     'mathtext.fontset':'dejavusans', 'svg.hashsalt':'attention-handbook-diagrams'})
TEXT_EXTENTS = []


def canvas(title, subtitle, height=1120):
    fig, ax = plt.subplots(figsize=(18, height/100), dpi=100)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0,1800), ylim=(height,0));ax.axis('off')
    text(ax, 60, 48, title, 27, weight='bold')
    text(ax, 60, 92, subtitle, 14, color=C['muted'])
    return fig, ax


def text(ax, x, y, label, size=14, color=None, ha='left', weight='normal'):
    artist=ax.text(x,y,label,fontsize=size,color=color or C['ink'],ha=ha,
                   va='center',fontweight=weight,linespacing=1.4)
    TEXT_EXTENTS.append((ax, artist))
    return artist


def box(ax,x,y,w,h,label='',fill='pale',edge='blue',size=14,weight='normal',z=2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=12',
                 linewidth=1.5,facecolor=C.get(fill,fill),edgecolor=C.get(edge,edge),zorder=z))
    if label:text(ax,x+w/2,y+h/2,label,size,ha='center',weight=weight)


def arrow(ax,points,color='line',lw=1.8,dashed=False):
    color=C.get(color,color)
    for a,b in zip(points[:-2],points[1:-1]):
        ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=lw,
                ls='--' if dashed else '-',zorder=1)
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',
                 mutation_scale=14,color=color,lw=lw,linestyle='--' if dashed else '-',zorder=1))


def panel(ax,x,y,w,h,title,subtitle=''):
    box(ax,x,y,w,h,fill='#fbfcfe',edge='border',z=0)
    text(ax,x+22,y+34,title,20,weight='bold')
    if subtitle:text(ax,x+22,y+72,subtitle,13,color=C['muted'])


def save(fig,ax,name,description):
    # Catch clipped labels before exporting; visual review checks wires and box placement.
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=ax.get_window_extent(renderer)
    for aa,artist in TEXT_EXTENTS:
        if aa is not ax:continue
        b=artist.get_window_extent(renderer)
        assert b.x0>=bounds.x0 and b.x1<=bounds.x1 and b.y0>=bounds.y0 and b.y1<=bounds.y1,artist.get_text()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT/(name+'.png'),dpi=150,facecolor='white',metadata={'Description':description})
    fig.savefig(OUT/(name+'.svg'),facecolor='white',metadata={'Date':None,'Description':description})
    svg=OUT/(name+'.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)


def head_sharing():
    fig,ax=canvas('MHA / MQA / GQA: what is shared?',
        '[To be reviewed]  Same four query heads in every panel; each KV tile represents its entire cached token history.',1100)
    colors=['#2669a8','#21856d','#a76a0b','#7953a6']
    designs=[('MHA',4,'One KV head per query head','4 KV heads | relative payload 100%'),
             ('MQA',1,'All queries share one KV head','1 KV head | relative payload 25%'),
             ('GQA',2,'Two queries share each KV head','2 KV heads | relative payload 50%')]
    for col,(title,hkv,sub,foot) in enumerate(designs):
        x=45+col*580;panel(ax,x,145,550,825,title,sub)
        centers=[x+70+i*136 for i in range(4)]
        kvcenters={4:centers,1:[x+274],2:[x+138,x+410]}[hkv]
        group=[a//(4//hkv) for a in range(4)]
        text(ax,x+22,265,'Stored K / V heads',14,weight='bold')
        for k,xx in enumerate(kvcenters):
            box(ax,xx-53,300,106,76,f'K{k} | V{k}\nall T tokens',fill='cache',edge=colors[k],size=13)
        for a,xx in enumerate(centers):
            k=group[a];source=kvcenters[k];wire=colors[k]
            arrow(ax,[(source,376),(source,403+k*12),(xx+39,403+k*12),(xx+39,580)],color=wire,lw=1.65)
            arrow(ax,[(xx-15,524),(xx-15,580)],color='blue')
            arrow(ax,[(xx,654),(xx,701)],color='blue')
            arrow(ax,[(xx,757),(xx,807),(x+274,807),(x+274,832)],color='line')
            box(ax,xx-35,466,70,58,f'Q{a}',size=17,weight='bold')
            box(ax,xx-49,580,98,74,f'Attn {a}',fill='#ffffff',edge=wire,size=14)
            box(ax,xx-38,701,76,56,f'O{a}',fill='mint',edge='green',size=15)
        box(ax,x+37,832,476,66,'Concat(O0, O1, O2, O3) → W_O',fill='#eef1f5',edge='line',size=15)
        text(ax,x+274,934,foot,14,ha='center',weight='bold')
    text(ax,60,1012,'Shared K/V does not imply identical attention weights: each Q head still computes its own softmax.',15)
    text(ax,60,1055,'Payload ratios assume equal T, head width and dtype. Projection details, masks and batch axes are omitted.',13,color=C['muted'])
    save(fig,ax,'attention-head-sharing','Four-query-head MHA, MQA and GQA; KV groups fan out to separate attention heads. No query heads are removed.')


def mla():
    fig,ax=canvas('MLA: cache a latent, read it with head-specific queries',
        '[To be reviewed]  DeepSeek-V2-style decoupled RoPE; conceptual absorbed decode path, one layer shown.',1260)
    panel(ax,45,145,620,995,'1  Write historical token j','Repeat for every processed token and layer')
    panel(ax,700,145,1055,995,'2  Read the stored history for new query i','Head a has its own content query, positional query and attention weights')
    box(ax,165,265,380,68,r'Historical hidden state $h_j$',size=18)
    arrow(ax,[(355,333),(355,362),(210,362),(210,398)],color='blue')
    arrow(ax,[(355,362),(510,362),(510,398)],color='blue')
    box(ax,80,398,260,88,'Down-project\nW_DKV',size=16)
    box(ax,380,398,260,88,'Position projection\nW_KR + RoPE(j)',size=16)
    arrow(ax,[(210,486),(210,620)],color='purple')
    arrow(ax,[(510,486),(510,620)],color='purple')
    box(ax,75,535,570,268,fill='cache',edge='purple',z=0)
    text(ax,360,570,'PERSISTENT CACHE',16,color=C['purple'],ha='center',weight='bold')
    box(ax,100,620,230,82,r'Latent $c_j$'+'\nwidth d_c',fill='#ffffff',edge='purple',size=17)
    box(ax,370,620,250,82,r'Positional key $k_j^R$'+'\nwidth d_r',fill='#ffffff',edge='purple',size=17)
    text(ax,360,755,'Store d_c + d_r elements / token',16,ha='center',weight='bold')
    box(ax,80,850,550,205,'No full per-head K/V cache required\nby this absorbed execution path.\n\nThe same latent supplies content keys\nand values; it is not stored twice.',fill='#ffffff',edge='border',size=15)
    # Lanes expose the positional term instead of rotating the compressed content state.
    for x,label in [(740,r'Read $c_1,\ldots,c_P$'),(1290,r'Read $k_1^R,\ldots,k_P^R$')]:
        box(ax,x,260,420,64,label,fill='cache',edge='purple',size=17)
    box(ax,740,365,420,88,r'$\widetilde q_{i,a}=(U_a^K)^T q^C_{i,a}$'+'\ntransform the content query',size=16)
    box(ax,1290,365,420,88,r'$q^R_{i,a}$'+'\nseparate rotated query',size=16)
    for x in [740,1290]:
        arrow(ax,[(x+210,324),(x+210,342),(x+440,342),(x+440,533),(x+420,533)],color='purple')
        arrow(ax,[(x+170,453),(x+170,495)],color='blue')
    box(ax,740,495,420,76,r'Content score: $\widetilde q_{i,a}^{T}c_j$',fill='#ffffff',edge='blue',size=17)
    box(ax,1290,495,420,76,r'Position score: $(q^R_{i,a})^T k_j^R$',fill='#ffffff',edge='blue',size=17)
    for x in [950,1500]:arrow(ax,[(x,571),(x,616),(1225,616),(1225,660)],color='blue')
    box(ax,865,660,720,91,'Add scores, scale, softmax over history j\n'+r'$\alpha_{ij,a}=\mathrm{softmax}_j(\mathrm{score}_{ij,a})$',size=17)
    arrow(ax,[(1225,751),(1225,808)],color='blue')
    box(ax,865,808,720,91,r'$u_{i,a}=\sum_j\alpha_{ij,a}c_j$'+'\nweighted read in latent coordinates',fill='cache',edge='purple',size=18)
    text(ax,800,828,'read c_j',12,color=C['purple'],ha='center')
    arrow(ax,[(805,324),(723,324),(723,852),(865,852)],color='purple',dashed=True)
    arrow(ax,[(1225,899),(1225,954)],color='blue')
    box(ax,865,954,720,112,r'$U_a^V u_{i,a}$'+' → concatenate heads → output projection\nValue up-projection can be composed with W_O.',fill='mint',edge='green',size=16)
    text(ax,60,1180,'MLA compresses features per token; this diagram still reads the full permitted history.',16,weight='bold')
    text(ax,60,1220,'Query generation/compression, masks and exact dimensions are simplified; see the accompanying equations.',13,color=C['muted'])
    save(fig,ax,'attention-mla','MLA writes one KV latent plus a shared positional key per token; head-specific absorbed content queries and separate RoPE scores produce a latent weighted read.')


def token_strip(ax,x,y,n,w,highlight=None,height=38):
    highlight=set(range(n)) if highlight is None else set(highlight)
    for i in range(n):
        active=i in highlight
        ax.add_patch(Rectangle((x+i*w,y),w-4,height,facecolor=C['mint'] if active else '#f0f2f5',
                     edgecolor=C['green'] if active else C['border'],lw=1.2,zorder=2))


def nsa():
    fig,ax=canvas('NSA: three complementary reads of the causal past',
        '[To be reviewed]  Native Sparse Attention: learned summaries + selected detailed blocks + recent local context.',1210)
    box(ax,280,150,1240,68,'Causal hidden-state history → branch-specific K/V projections',fill='#eef1f5',edge='line',size=18)
    for x in [325,900,1475]:arrow(ax,[(900,218),(900,245),(x,245),(x,290)],color='line')
    for x,title,sub in [(45,'COMPRESSION','Coarse global context'),(620,'SELECTION','Fine-grained distant details'),(1195,'SLIDING WINDOW','Recent local context')]:
        panel(ax,x,290,560,585,title,sub)
    text(ax,325,408,'Learned block compression',16,ha='center',weight='bold')
    token_strip(ax,121,451,6,70)
    arrow(ax,[(325,489),(325,555)],color='green')
    box(ax,95,555,460,102,r'Attn($q_t$, summaries)'+'\nproduces compression weights p_cmp',fill='mint',edge='green',size=15)
    arrow(ax,[(325,657),(325,758)],color='green')
    box(ax,165,758,320,66,r'$o_t^{\mathrm{cmp}}$',fill='mint',edge='green',size=20)
    box(ax,670,406,460,97,'Derive block importance scores\nAggregate query heads within KV group',fill='mint',edge='green',size=14)
    # The dashed wire is selection guidance, not the compression branch output.
    arrow(ax,[(555,605),(588,605),(588,454),(670,454)],color='purple',dashed=True)
    text(ax,628,436,'weights',11,color=C['purple'],ha='center')
    arrow(ax,[(900,503),(900,548)],color='green')
    box(ax,710,548,380,63,'Select top-n blocks',fill='mint',edge='green',size=16)
    arrow(ax,[(900,611),(900,642)],color='green')
    token_strip(ax,678,642,8,56,highlight=[1,2,6])
    text(ax,900,708,'Read original token K/V in selected blocks',13,ha='center',color=C['muted'])
    arrow(ax,[(900,680),(650,680),(650,735),(900,735),(900,758)],color='green')
    box(ax,670,758,460,66,r'Attn($q_t$, selected K/V) → $o_t^{\mathrm{slc}}$',fill='mint',edge='green',size=15)
    token_strip(ax,1255,451,8,56,highlight=[4,5,6,7])
    text(ax,1475,418,'Keep a recent local window',15,ha='center',weight='bold')
    arrow(ax,[(1475,489),(1475,555)],color='green')
    box(ax,1245,555,460,102,r'Attn($q_t$, recent K/V)'+'\nlocal patterns have a separate path',fill='mint',edge='green',size=15)
    arrow(ax,[(1475,657),(1475,758)],color='green')
    box(ax,1315,758,320,66,r'$o_t^{\mathrm{win}}$',fill='mint',edge='green',size=20)
    for x,label in [(325,'g_cmp'),(900,'g_slc'),(1475,'g_win')]:
        arrow(ax,[(x,824),(x,915),(900,915),(900,956)],color='green')
        text(ax,x+13,894,'× '+label,14,color=C['green'])
    box(ax,460,956,880,85,r'$o_t=g_t^{\mathrm{cmp}}o_t^{\mathrm{cmp}}+g_t^{\mathrm{slc}}o_t^{\mathrm{slc}}+g_t^{\mathrm{win}}o_t^{\mathrm{win}}$',fill='pale',edge='blue',size=22)
    text(ax,900,1080,'Gates = sigmoid(MLP(input)); they need not sum to 1. Every attention branch uses the current query q_t.',14,ha='center')
    text(ax,60,1140,'Dashed arrow: selection guidance. Compression and selection blocks need not have identical geometry.',13,color=C['muted'])
    text(ax,60,1180,'Diagram is schematic: causal boundaries, overlapping blocks and group/kernel layout follow the trained architecture.',13,color=C['muted'])
    save(fig,ax,'attention-nsa','NSA uses independent branch K/V: learned compressed summaries guide block selection; fine-grained selected attention and a local-window path combine through sigmoid gates.')


def prefill():
    fig,ax=canvas('Prefill → decode: when does a token enter the KV cache?',
        '[To be reviewed]  Ordinary single-token causal generation; three-token prompt, one sequence, layerwise state.',1220)
    for x,label,color in [(60,'Known input / computation','blue'),(690,'Newly sampled output','amber'),(1280,'Persistent processed-token KV','purple')]:
        box(ax,x,144,440,48,label,fill='pale' if color=='blue' else 'sand' if color=='amber' else 'cache',edge=color,size=13)
    rows=[(285,'PREFILL','x0, x1, x2','Process all 3 prompt tokens\nCausal pass through every layer','y1',['x0','x1','x2'],3),
          (585,'DECODE 1','y1','Read 3 cached positions\nProcess y1; append its K/V','y2',['x0','x1','x2','y1'],4),
          (885,'DECODE 2','y2','Read 4 cached positions\nProcess y2; append its K/V','y3',['x0','x1','x2','y1','y2'],5)]
    for row,(y,stage,inp,model,out,tokens,count) in enumerate(rows):
        text(ax,60,y-58,stage,18,weight='bold')
        box(ax,60,y,260,84,inp,size=21,weight='bold')
        box(ax,405,y-10,445,104,model,size=16)
        box(ax,945,y,365,84,'Logits → sample '+out,fill='sand',edge='amber',size=18)
        arrow(ax,[(320,y+42),(405,y+42)],color='blue')
        arrow(ax,[(850,y+42),(945,y+42)],color='blue')
        box(ax,1390,y-10,355,137,fill='cache',edge='purple')
        text(ax,1567,y+14,'CACHE: '+str(count)+' POSITIONS',14,ha='center',color=C['purple'],weight='bold')
        n=len(tokens);w=58;start=1567-n*w/2
        for j,tok in enumerate(tokens):box(ax,start+j*w,y+49,w-5,47,tok,fill='#ffffff',edge='purple',size=13)
        # Write path routes below the sampling box: selecting a token is not itself a KV write.
        arrow(ax,[(627,y+94),(627,y+150),(1567,y+150),(1567,y+127)],color='purple')
        text(ax,650,y+171,'Store K/V for the tokens just processed',12,color=C['purple'])
        if row<2:
            arrow(ax,[(1127,y+84),(1127,y+224),(300,y+224),(300,y+300)],color='amber',dashed=True)
            text(ax,710,y+244,'Feed sampled '+out+' into the next forward pass',13,color=C['amber'],ha='center')
            arrow(ax,[(1730,y+127),(1770,y+127),(1770,y+290),(1745,y+290)],color='purple',dashed=True)
        else:
            text(ax,1130,y+220,'Continue unless EOS / stopping rule ends generation.',13,color=C['amber'],ha='center')
    text(ax,60,1147,'After sampling y1, the cache still contains only x0–x2. y1 gets K/V when the next pass processes y1.',15,weight='bold')
    text(ax,60,1190,'Each layer owns its own cache. Earlier causal K/V are reused; old queries are not needed for future attention.',13,color=C['muted'])
    save(fig,ax,'prefill-decode-kv-cache','Prefill processes a three-token prompt and predicts y1 with three cached positions; subsequent passes process y1 then y2, append their KV, and predict y2 then y3.')


if __name__=='__main__':
    head_sharing();mla();nsa();prefill()
    print('Generated four PNG previews and four editable SVG diagrams in figs/.')
