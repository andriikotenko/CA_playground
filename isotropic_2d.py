import itertools

import numpy as np
import numba as nb

import pygame as pg

FPS = 60
FPS = 0

WINDOW_SIZE = (800,800)

# ARRAY_SHAPE = (10,10)
ARRAY_SHAPE = (80,80)
# ARRAY_SHAPE = (400,400)


# EDGE_CONDITIONS = "zero"
# EDGE_CONDITIONS = "periodical"

CELL_SIZE = min(WINDOW_SIZE)/max(ARRAY_SHAPE)

cell_arr = np.random.rand(*ARRAY_SHAPE)>0.4
# cell_arr = np.diagflat( np.ones(ARRAY_SHAPE[0],dtype=bool) ) + np.diagflat( np.ones(ARRAY_SHAPE[0],dtype=bool) )[::-1]

cell_arr_updated = cell_arr.copy()

@nb.njit()
def rule_result(cell_value,alive_neighbourhood):
    if cell_value:
        return (alive_neighbourhood==2) or (alive_neighbourhood==3)
    else:
        return alive_neighbourhood==3


@nb.njit()
def apply_rule_bulk(cells, cells_updated):
    alive_neighbourhood = int(-1)
    for i in range(1,cells.shape[0]-1):
        for j in range(1,cells.shape[1]-1):
            alive_neighbourhood = 0
            for sh_hor in (-1,0,1):
                for sh_vert in (-1,0,1):
                    alive_neighbourhood+=cells[i+sh_hor,j+sh_vert]
            alive_neighbourhood-=cells[i,j]
            cells_updated[i,j] = rule_result(cells[i,j], alive_neighbourhood)


def update_cells():
    apply_rule_bulk(cell_arr, cell_arr_updated)

    # if EDGE_CONDITIONS=="zero":
    #     raise NotImplementedError
    # elif EDGE_CONDITIONS=="peridical":
    #     raise NotImplementedError
    # else:
    #     raise RuntimeError

    np.copyto(cell_arr, cell_arr_updated)


pg.init()

screen = pg.display.set_mode(WINDOW_SIZE)
clock = pg.time.Clock()

def check_events():
    for event in pg.event.get():
        if event.type==pg.QUIT or (event.type==pg.KEYDOWN and event.key==pg.K_ESCAPE):
            pg.quit()

def update():
    update_cells()

    pg.display.flip()
    clock.tick(FPS)
    pg.display.set_caption("fps = {}".format(clock.get_fps()))

def draw():
    screen.fill("black")
    draw_cells()

def draw_cells():
    for i,j in itertools.product( range(ARRAY_SHAPE[0]), range(ARRAY_SHAPE[1]) ):
        if cell_arr[i,j]==False:
            continue
        pg.draw.rect(screen,
            "white",
            (CELL_SIZE*i, CELL_SIZE*j,
            CELL_SIZE, CELL_SIZE))

if __name__=="__main__":
    while True:
        check_events()
        draw()
        update()