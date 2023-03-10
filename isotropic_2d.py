import itertools

import numpy as np
import numba as nb

import pygame as pg


### TODO:
# 1) implement few other rules (and compare speed)
# 2) GPU processing (mostly for future 3d explorations)
# 3) check that i really pass references and not copying arrays with '='. Also understand variable scopes
# 4) center if not square screen?
# 5) compare simulation speed when 1. using np.copyto, 2. when switching references of arrays
# 6) decouple FPS from simulation speed and interaction speed (responsiveness)



FPS = 60
FPS = 0

WINDOW_SIZE = (600,600)

CELL_SIZE = 3

ARRAY_SHAPE = (WINDOW_SIZE[0]//CELL_SIZE, WINDOW_SIZE[1]//CELL_SIZE)

# EDGE_CONDITIONS = "zero"
EDGE_CONDITIONS = "periodical"


cell_arr_a = np.pad(np.random.rand(*ARRAY_SHAPE)>0.4, 1)
# cell_arr = np.diagflat( np.ones(ARRAY_SHAPE[0],dtype=bool) ) + np.diagflat( np.ones(ARRAY_SHAPE[0],dtype=bool) )[::-1]

cell_arr_b = cell_arr_a.copy()


NOW_ARR_A = True

cell_arr = cell_arr_a
cell_arr_updated = cell_arr_b


SIMULATION_ON = True

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
            cells_updated[i,j] = rule_result(cells[i,j], n_alive_neighbours)

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


if EDGE_CONDITIONS=="zero":
    set_pad = set_pad_zero
elif EDGE_CONDITIONS=="periodical":
    set_pad = set_pad_periodical
else:
    raise RuntimeError

def update_cells():
    global NOW_ARR_A
    global cell_arr
    global cell_arr_updated

    set_pad(cell_arr)

    apply_rule(cell_arr, cell_arr_updated)

    if NOW_ARR_A:
        cell_arr = cell_arr_a
        cell_arr_updated = cell_arr_b
    else:
        cell_arr = cell_arr_b
        cell_arr_updated = cell_arr_a
    NOW_ARR_A = not NOW_ARR_A

def toggle_cell_state(pos):
    index = (pos[0]//CELL_SIZE+1, pos[1]//CELL_SIZE+1)
    cell_arr[index] = not cell_arr[index]
    if cell_arr[index]:
        draw_cell(*index)


pg.init()

screen = pg.display.set_mode(WINDOW_SIZE)
clock = pg.time.Clock()

def check_events():
    for event in pg.event.get():
        if event.type==pg.QUIT or (event.type==pg.KEYDOWN and event.key==pg.K_ESCAPE):
            pg.quit()
            exit()
        elif event.type == pg.MOUSEBUTTONDOWN:
            toggle_cell_state(event.pos)
        elif event.type==pg.KEYDOWN and event.key==pg.K_SPACE:
            global SIMULATION_ON
            SIMULATION_ON = not SIMULATION_ON


def update():
    if SIMULATION_ON:
        update_cells()
        caption = "fps = {}".format(clock.get_fps())
    else:
        caption = "PAUSED"
        pg.display.set_caption(caption)

    pg.display.flip()
    clock.tick(FPS)

    pg.display.set_caption(caption)


def draw():
    screen.fill("black")
    draw_cells()

def draw_cell(i,j):
    pg.draw.rect(screen,
        "white",
        (CELL_SIZE*i, CELL_SIZE*j, CELL_SIZE, CELL_SIZE))

def draw_cells():
    for i,j in itertools.product( range(ARRAY_SHAPE[0]), range(ARRAY_SHAPE[1]) ):
        if cell_arr[i+1,j+1]==False:
            continue
        draw_cell(i,j)

if __name__=="__main__":
    while True:
        check_events()
        draw()
        update()