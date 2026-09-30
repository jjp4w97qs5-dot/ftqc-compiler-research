from __future__ import annotations

"""2D transverse-field Ising kernelの格子構造とProgram builder。"""

from dataclasses import dataclass

from ..program_ir import Block, Op, Program


@dataclass(frozen=True)
class Ising2DWorkload:
    """2D Ising評価で使用する格子・Trotter step・回転角。"""

    height: int = 10
    width: int = 10
    steps: int = 8
    theta_h: float = 0.30
    theta_j: float = 0.20

    @property
    def qubits(self) -> int:
        # 格子全体のlogical qubit数を返す。
        return self.height * self.width

    @property
    def edges(self) -> int:
        # 水平・垂直最近接edgeの総数を返す。
        return self.height * (self.width - 1) + (self.height - 1) * self.width


def qid(r: int, c: int, w: int) -> int:
    """格子座標をrow-majorのqubit IDへ変換する。"""

    return r * w + c


def qname(r: int, c: int, w: int) -> str:
    """格子座標に対応するProgram IR上のqubit名を返す。"""

    return f"q{qid(r,c,w)}"


def grid_edges(wl: Ising2DWorkload) -> list[tuple[int, int, str]]:
    """既存順序で水平edge、続いて垂直edgeを列挙する。"""

    out: list[tuple[int, int, str]] = []
    for r in range(wl.height):
        for c in range(wl.width - 1):
            out.append((qid(r,c,wl.width), qid(r,c+1,wl.width), "horizontal"))
    for r in range(wl.height - 1):
        for c in range(wl.width):
            out.append((qid(r,c,wl.width), qid(r+1,c,wl.width), "vertical"))
    return out


def edge_color(u: int, v: int, width: int) -> int:
    """最近接edgeを従来の4 matching colorへ分類する。"""

    r0,c0=divmod(u,width); r1,c1=divmod(v,width)
    return c0 % 2 if r0 == r1 else 2 + (r0 % 2)


def build_ising2d_program(wl: Ising2DWorkload) -> Program:
    """field層と4色interaction層から2D Ising Programを構築する。"""

    names=[qname(r,c,wl.width) for r in range(wl.height) for c in range(wl.width)]
    edges=grid_edges(wl)
    body=[]
    for step in range(wl.steps):
        field=[]
        for q in names:
            meta={"phase":"field","trotter_step":step}
            field += [Op("H",(q,),meta=meta),Op("RZ",(q,),(wl.theta_h,),meta=meta),Op("H",(q,),meta=meta)]
        body.append(Block("field_layer",f"field_{step}",field,{"trotter_step":step}))
        for color in range(4):
            block=[]
            for pair_index,(u,v,orientation) in enumerate(e for e in edges if edge_color(e[0],e[1],wl.width)==color):
                a,b=names[u],names[v]
                meta={"phase":"interaction","trotter_step":step,"edge_color":color,"orientation":orientation,"pair_index":pair_index}
                block += [Op("CX",(a,b),meta=meta),Op("RZ",(b,),(wl.theta_j,),meta=meta),Op("CX",(a,b),meta=meta)]
            body.append(Block("interaction_layer",f"interaction_{step}_{color}",block,{"trotter_step":step,"edge_color":color}))
    return Program(f"tfim2d_{wl.height}x{wl.width}_steps{wl.steps}",names,body)
