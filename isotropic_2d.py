import itertools

import numpy as np
import numba as nb

import pygame as pg


### TODO:
# 1) interactivity: mouse cell state toggle
# 2) check thoroughly periodical edge conditions using 1)
# 3) implement few other rules (and compare speed)
# 4) GPU processing (mostly for future 3d explorations)



FPS = 60
FPS = 0

WINDOW_SIZE = (600,600)

CELL_SIZE = 10

ARRAY_SHAPE = (WINDOW_SIZE[0]//CELL_SIZE, WINDOW_SIZE[1]//CELL_SIZE)

# EDGE_CONDITIONS = "zero"
EDGE_CONDITIONS = "periodical"


cell_arr = np.pad(np.random.rand(*ARRAY_SHAPE)>0.4, 1)
# cell_arr = np.diagflat( np.ones(ARRAY_SHAPE[0],dtype=bool) ) + np.diagflat( np.ones(ARRAY_SHAPE[0],dtype=bool) )[::-1]

cell_arr_updated = cell_arr[1:-1,1:-1].copy()


@nb.njit()
def rule_result(cell_value,alive_neighbourhood):
    if cell_value:
        return (alive_neighbourhood==2) or (alive_neighbourhood==3)
    else:
        return alive_neighbourhood==3

@nb.njit()
def get_n_alive_neighbours(cells, i, j):
    alive_neighbourhood = int(0)
    for sh_hor in (-1,0,1):
        for sh_vert in (-1,0,1):
            alive_neighbourhood+=cells[i+sh_hor,j+sh_vert]
    alive_neighbourhood-=cells[i,j]
    return alive_neighbourhood

@nb.njit()
def apply_rule(cells, cells_updated):
    for i in range(1,cells.shape[0]-1):
        for j in range(1,cells.shape[1]-1):
            n_alive_neighbours = get_n_alive_neighbours(cells, i, j)
            cells_updated[i-1,j-1] = rule_result(cells[i,j], n_alive_neighbours)

@nb.njit()
def set_pad_zero(cells):
    for i in range(cells.shape[0]):
        cells[i,0]=0
        cells[i,-1]=0
    for j in range(cells.shape[1]):
        cells[0,j]=0
        cells[-1,j]=0


@nb.njit()
def set_pad_periodical_1d(cells_1d):
    cells_1d[0]=cells_1d[-2]
    cells_1d[-1]=cells_1d[1]

@nb.njit()
def set_pad_periodical_2d(cells_2d):
    for i in range(1,cells_2d.shape[0]-1):
        set_pad_periodical_1d(cells_2d[i,:])
    for j in range(1,cells_2d.shape[1]-1):
        set_pad_periodical_1d(cells_2d[:,j])
    cells_2d[0,0]=cells_2d[-2,-2]
    cells_2d[-1,-1]=cells_2d[1,1]
    cells_2d[0,-1]=cells_2d[-2,1]
    cells_2d[-1,0]=cells_2d[1,-2]

set_pad_periodical = set_pad_periodical_2d

# @nb.njit()
# def set_pad_periodical(cells):
#     for i in range(1,cells.shape[0]-1):
#         cells[i,0]=cells[i,-2]
#         cells[i,-1]=cells[i,1]
#     for j in range(1,cells.shape[1]-1):
#         cells[0,j]=cells[-2,j]
#         cells[-1,j]=cells[1,j]
#     cells[0,0]=cells[-2,-2]
#     cells[-1,-1]=cells[1,1]
#     cells[0,-1]=cells[-2,1]
#     cells[-1,0]=cells[1,-2]



if EDGE_CONDITIONS=="zero":
    set_pad = set_pad_zero
elif EDGE_CONDITIONS=="periodical":
    set_pad = set_pad_periodical
else:
    raise RuntimeError

def update_cells():
    set_pad(cell_arr)

    apply_rule(cell_arr, cell_arr_updated)

    np.copyto(cell_arr[1:-1,1:-1], cell_arr_updated)


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
        if cell_arr[i+1,j+1]==False:
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