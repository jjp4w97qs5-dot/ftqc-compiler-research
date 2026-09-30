from __future__ import annotations

"""2D Isingの純粋なtile配置・edge順序・boundary round計画。"""

from ..programs.ising2d import Ising2DWorkload, grid_edges, qid
from ..sam_layout import sam_coord


def tile_cr_for_qubit(q: int, wl: Ising2DWorkload) -> int:
    """qubitの2×2 quadrantに対応するCR IDを返す。"""

    r,c=divmod(q,wl.width)
    return (2 if r >= wl.height//2 else 0) + (1 if c >= wl.width//2 else 0)


def qubits_for_tile(wl: Ising2DWorkload, cr: int) -> list[int]:
    """指定tileのqubitを既存のserpentine順で返す。"""

    sites=[q for q in range(wl.qubits) if tile_cr_for_qubit(q,wl)==cr]
    rows: dict[int,list[int]]={}
    for q in sites:
        r,c=divmod(q,wl.width); rows.setdefault(r,[]).append(c)
    out=[]
    for r in sorted(rows):
        cols=sorted(rows[r],reverse=(r%2==1))
        out.extend(qid(r,c,wl.width) for c in cols)
    return out


def tile_initial_layout(
    wl: Ising2DWorkload,
    cr_count: int,
    sam_type: str,
) -> dict[int, tuple[str, int, int, int]]:
    """各tileを同番号bankへ置く既存の初期SAM配置を返す。"""

    loc: dict[int, tuple[str, int, int, int]] = {}
    for cr in range(cr_count):
        for index, q in enumerate(qubits_for_tile(wl, cr)):
            row, col = sam_coord(sam_type, index)
            loc[q] = ("SAM", cr, row, col)
    return loc


def classify_edges(wl: Ising2DWorkload) -> tuple[dict[int,list[tuple[int,int,str]]],list[tuple[int,int,str]]]:
    """格子edgeをtile内部edgeとtile境界edgeへ分ける。"""

    internal={cr:[] for cr in range(4)}; boundary=[]
    for e in grid_edges(wl):
        a,b,_=e
        if tile_cr_for_qubit(a,wl)==tile_cr_for_qubit(b,wl): internal[tile_cr_for_qubit(a,wl)].append(e)
        else: boundary.append(e)
    return internal,boundary


def edge_center(e: tuple[int,int,str], width:int) -> tuple[float,float]:
    """edge中心の格子座標を返す。"""

    a,b,_=e; r0,c0=divmod(a,width); r1,c1=divmod(b,width)
    return ((r0+r1)/2,(c0+c1)/2)


def order_tile_edges_for_reuse(edges: list[tuple[int,int,str]], width:int) -> list[tuple[int,int,str]]:
    """共有endpointと近接性を優先してtile内部edgeを並べる。"""

    # 1量子ビットだけのtileには内部edgeがない。
    if not edges:
        return []
    remaining=set(edges); out=[]
    current=min(remaining)
    while remaining:
        if current not in remaining:
            cr,cc=edge_center(out[-1],width)
            current=min(remaining,key=lambda e:(abs(edge_center(e,width)[0]-cr)+abs(edge_center(e,width)[1]-cc),e))
        out.append(current); remaining.remove(current)
        if not remaining: break
        current_set={current[0],current[1]}; cr,cc=edge_center(current,width)
        current=min(remaining,key=lambda e:(0 if current_set & {e[0],e[1]} else 1,abs(edge_center(e,width)[0]-cr)+abs(edge_center(e,width)[1]-cc),e))
    return out


def schedule_boundary_edge_rounds(edges: list[tuple[int,int,str]], wl: Ising2DWorkload) -> list[list[tuple[int,tuple[int,int,str]]]]:
    """qubit・CR競合のない既存順のboundary roundを構築する。"""

    rounds: list[list[tuple[int,tuple[int,int,str]]]]=[]
    for index,e in enumerate(sorted(edges)):
        owners=[tile_cr_for_qubit(e[0],wl),tile_cr_for_qubit(e[1],wl)]
        if index%2: owners.reverse()
        placed=False
        for round_ in rounds:
            used_cr={cr for cr,_ in round_}; used_q={q for _,x in round_ for q in x[:2]}
            for owner in owners:
                if owner not in used_cr and not ({e[0],e[1]} & used_q):
                    round_.append((owner,e)); placed=True; break
            if placed: break
        if not placed:
            rounds.append([(owners[0],e)])
    return rounds
