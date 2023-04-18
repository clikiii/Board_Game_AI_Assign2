import numpy as np
import copy
import time
import psutil
import asyncio
import random

from colors import colors

def timing(func):
    def wrapper(*args, **kwargs):
        print(colors.PURPLE, '\n-----Called-----', colors.ENDC)
        start = time.time()
        ret = func(*args,**kwargs)
        end = time.time()
        print(colors.PURPLE, '{0}: {1}s\n'.format(func.__name__, end - start), colors.ENDC)
        print(colors.PURPLE, psutil.Process().memory_info().rss / 1024 ** 2, len(TRANS_TABLE), colors.ENDC)
        return ret
    return wrapper


class Constant:
    POS_INF = np.inf
    NEG_INF = -np.inf
    WOLF_MAX = 2
    SHEEP_MIN = 1
    EMPTY_CELL = 0
    MIN_DEPTH = 3
    MAX_DEPTH = 7


class Move:
    def __init__(self, start_pos: list, end_pos: list, is_eat: bool) -> None:
        self.start_pos = {'row': start_pos[0], 'col': start_pos[1]}
        self.end_pos = {'row': end_pos[0], 'col': end_pos[1]}
        self.is_eat = is_eat

    def __str__(self) -> str:
        return '[({0}, {1}), ({2}, {3})]'.format(
            self.start_pos['row'], self.start_pos['col'], self.end_pos['row'], self.end_pos['col']
        )
    
    def tolist(self) -> list:
        return [self.start_pos['row'], self.start_pos['col'], self.end_pos['row'], self.end_pos['col']]


def get_valid(matrix: np.ndarray, start_pos: tuple) -> list:
    """
    Returns: list[Move]
    """
    start_pos = list(start_pos)
    moves = []
    if matrix[start_pos[0]][start_pos[1]] == Constant.EMPTY_CELL: return moves # secure

    if start_pos[0]-1 >=0 and matrix[start_pos[0]-1][start_pos[1]] == Constant.EMPTY_CELL:
        moves.append(Move(start_pos, [start_pos[0]-1, start_pos[1]], False))

    if start_pos[0]+1 <=4 and matrix[start_pos[0]+1][start_pos[1]] == Constant.EMPTY_CELL:
        moves.append(Move(start_pos, [start_pos[0]+1, start_pos[1]], False))

    if start_pos[1]-1 >=0 and matrix[start_pos[0]][start_pos[1]-1] == Constant.EMPTY_CELL:
        moves.append(Move(start_pos, [start_pos[0], start_pos[1]-1], False))

    if start_pos[1]+1 <=4 and matrix[start_pos[0]][start_pos[1]+1] == Constant.EMPTY_CELL:
        moves.append(Move(start_pos, [start_pos[0], start_pos[1]+1], False))

    # print(colors.CYAN, "\n valid moves", moves, colors.ENDC)
    return moves

def get_eat(matrix: np.ndarray, start_pos: tuple) -> list:
    """
    Returns: list[Move]
    """
    start_pos = list(start_pos)
    moves = []
    if matrix[start_pos[0]][start_pos[1]] == Constant.EMPTY_CELL or matrix[start_pos[0]][start_pos[1]] == Constant.SHEEP_MIN: return moves # secure

    if (start_pos[0]-2 >=0
        and matrix[start_pos[0]-1][start_pos[1]] == Constant.EMPTY_CELL
        and matrix[start_pos[0]-2][start_pos[1]] == Constant.SHEEP_MIN):
        moves.append(Move(start_pos, [start_pos[0]-2, start_pos[1]], True))

    if (start_pos[0]+2 <=4
        and matrix[start_pos[0]+1][start_pos[1]] == Constant.EMPTY_CELL
        and matrix[start_pos[0]+2][start_pos[1]] == Constant.SHEEP_MIN):
        moves.append(Move(start_pos, [start_pos[0]+2, start_pos[1]], True))

    if (start_pos[1]-2 >=0 
        and matrix[start_pos[0]][start_pos[1]-1] == Constant.EMPTY_CELL
        and matrix[start_pos[0]][start_pos[1]-2] == Constant.SHEEP_MIN):
        moves.append(Move(start_pos, [start_pos[0], start_pos[1]-2], True))

    if (start_pos[1]+2 <=4 
        and matrix[start_pos[0]][start_pos[1]+1] == Constant.EMPTY_CELL
        and matrix[start_pos[0]][start_pos[1]+2] == Constant.SHEEP_MIN):
        moves.append(Move(start_pos, [start_pos[0], start_pos[1]+2], True))

    # print(colors.YELLO, "\n eat moves", moves, colors.ENDC)
    return moves


def get_moves_wolf(matrix: np.ndarray) -> list:
    """
    Returns: list[Move]
    """
    moves = []
    w_poses = list(zip(*np.where(matrix == Constant.WOLF_MAX))) # [(row, col), (0, 1), (1, 2), (1, 3)]
    for wp in w_poses: moves += get_eat(matrix, wp) # eat moves will be in the front of other moves
    random.shuffle(moves)

    tmp = []
    for wp in w_poses: tmp += get_valid(matrix, wp)
    random.shuffle(tmp)
    moves += tmp
    return moves

def get_moves_sheep(matrix: np.ndarray) -> list:
    """
    Returns: list[Move]
    """
    moves = []
    s_poses = list(zip(*np.where(matrix == Constant.SHEEP_MIN))) # [(row, col), (0, 1), (1, 2), (1, 3)]
    for sp in s_poses:
        moves += get_valid(matrix, sp)
    random.shuffle(moves)
    return moves


TRANS_TABLE = {}
class TransRec:
    def __init__(self, value = None, depth = -1, eat_mov_cnt = None, valid_mov_cnt = None) -> None:
        """
        depth:
            cur_node: d = 0
            child of cur: d = 1
            child of child of cur: d = 2
        """
        self.s_val = {'value': value, 'depth': depth}
        self.child_moves_cnt = {'eat_mov_cnt': eat_mov_cnt, 'valid_mov_cnt': valid_mov_cnt}

async def tidy_table(s_num):
    delk = []
    for k in TRANS_TABLE.keys():
        if int(k[0]) > s_num: delk.append(k)
    for k in delk: del TRANS_TABLE[k]

class MatrixNode:
    def __init__(self, matrix: np.ndarray, playing) -> None:
        """
        playing: the player will play the next move.
        """
        self.matrix = matrix
        self.playing = playing
        self.hash_str = (str(np.count_nonzero(self.matrix == Constant.SHEEP_MIN)) 
                        + str(hash(matrix.tobytes())) 
                        + str(playing)) # same matrix can have different current player

    def is_terminate(self) -> int:
        """
        determine the if the game ends.
        Returns:
            wolf wins - constant.wolf_max
            sheep wins - constant.sheep_min
            not end - 0
        """

        # check wolf win
        if np.count_nonzero(self.matrix == Constant.SHEEP_MIN) <= 2: return Constant.WOLF_MAX

        # check sheep win
        new_hstr = self.hash_str[:-1] + str(Constant.WOLF_MAX)
        if TRANS_TABLE.get(new_hstr) and (TRANS_TABLE.get(new_hstr)).child_moves_cnt:
            w_moves_cnt = sum((TRANS_TABLE.get(new_hstr)).child_moves_cnt.values())
        else: w_moves_cnt = len(get_moves_wolf(self.matrix))
        if w_moves_cnt == 0: return Constant.SHEEP_MIN

        # not end
        return 0

    def get_child_moves(self) -> list:
        """
        Returns: list[Move]
        """
        if TRANS_TABLE.get(self.hash_str) and (TRANS_TABLE.get(self.hash_str)).child_moves:
            return (TRANS_TABLE.get(self.hash_str)).child_moves

        child_moves = get_moves_wolf(self.matrix) if self.playing == Constant.WOLF_MAX else get_moves_sheep(self.matrix)
        
        TRANS_TABLE[self.hash_str] = TransRec(child_moves=child_moves)
        return child_moves
    
    def get_material_value(self):
        # NOTE: currently only consider self side
        value = 0
        # if a wolf is not captured, then +500
        if self.playing == Constant.WOLF_MAX:
            w_poses = list(zip(*np.where(self.matrix == Constant.WOLF_MAX)))
            for wp in w_poses:
                cnt = 0
                if wp[0]-1 >=0: cnt += (self.matrix[wp[0]-1][wp[1]] == Constant.EMPTY_CELL)
                if wp[0]+1 <=4: cnt += (self.matrix[wp[0]+1][wp[1]] == Constant.EMPTY_CELL)
                if wp[1]-1 >=0: cnt += (self.matrix[wp[0]][wp[1]-1] == Constant.EMPTY_CELL)
                if wp[1]+1 <=4: cnt += (self.matrix[wp[0]][wp[1]+1] == Constant.EMPTY_CELL)
                if cnt: value += 500

        # if a sheep is alive, then +100
        elif self.playing == Constant.SHEEP_MIN:
            s_num = np.count_nonzero(self.matrix == Constant.SHEEP_MIN)
            value += s_num * 100

        return value
    
    def get_mobility_value(self):
        value = 0

        # one valid step, +100
        # one eat step, +200
        if self.playing == Constant.WOLF_MAX:
            cur_player_child_moves = self.get_child_moves() # TODO: only record move_num for memory reason
            for mov in cur_player_child_moves:
                value += 200 if mov.is_eat else 100

        # if a sheep is near a wolf, +100
        # if a sheep can be eaten by a wolf, -200
        if self.playing == Constant.SHEEP_MIN:
            w_poses = list(zip(*np.where(self.matrix == Constant.WOLF_MAX)))
            for wp in w_poses:
                cnt = 0
                cnt += (wp[0]-1 <0 or (self.matrix[wp[0]-1][wp[1]] == Constant.SHEEP_MIN))
                cnt += (wp[0]+1 >4 or (self.matrix[wp[0]+1][wp[1]] == Constant.SHEEP_MIN))
                cnt += (wp[1]-1 <0 or (self.matrix[wp[0]][wp[1]-1] == Constant.SHEEP_MIN))
                cnt += (wp[1]+1 >4 or (self.matrix[wp[0]][wp[1]+1] == Constant.SHEEP_MIN))
                if cnt == 4: cnt += 5

                if wp[0]-2 >=0: cnt -= ((self.matrix[wp[0]-1][wp[1]] == Constant.EMPTY_CELL) and (self.matrix[wp[0]-2][wp[1]] == Constant.SHEEP_MIN))*2
                if wp[0]+2 <=4: cnt -= ((self.matrix[wp[0]+1][wp[1]] == Constant.EMPTY_CELL) and (self.matrix[wp[0]+2][wp[1]] == Constant.SHEEP_MIN))*2
                if wp[1]-2 >=0: cnt -= ((self.matrix[wp[0]][wp[1]-1] == Constant.EMPTY_CELL) and (self.matrix[wp[0]][wp[1]-2] == Constant.SHEEP_MIN))*2
                if wp[1]+2 <=4: cnt -= ((self.matrix[wp[0]][wp[1]+1] == Constant.EMPTY_CELL) and (self.matrix[wp[0]][wp[1]+2] == Constant.SHEEP_MIN))*2
                value += cnt*100
        
        return value
    
    def evaluate(self):
        # TODO: state and move evaluate together
        # TODO: 同伴间不会配合
        ret = self.is_terminate()

        # real terminal case
        if ret == Constant.WOLF_MAX: return 10000
        if ret == Constant.SHEEP_MIN: return -10000

        # vals
        material_val = self.get_material_value()
        mobility_val = self.get_mobility_value()

        # weights
        material_w = 0.8
        mobility_val = 1
        
        final_score = material_w * material_val + mobility_val * mobility_val

        return final_score if self.playing == Constant.WOLF_MAX else -final_score


def get_next_node(matrix_node: MatrixNode, move: Move):
    """
    a new matrix node after the move
    """
    ret = copy.deepcopy(matrix_node.matrix)
    ret[move.end_pos['row']][move.end_pos['col']] = matrix_node.playing
    ret[move.start_pos['row'], move.start_pos['col']] = Constant.EMPTY_CELL

    return MatrixNode(
        ret, 
        Constant.SHEEP_MIN if matrix_node.playing == Constant.WOLF_MAX else Constant.WOLF_MAX
    )


def simple_ab_search(matrix_node: MatrixNode, alpha, beta, depth):
    """
    wolf side moves first, the very first root is wolf node.
    wolf - max node
    sheep - min node

    matrix: just like matrix in the ai_algo function, a 2d list.
    
    Returns: value of evaluate fucntion
    """

    def get_max(matrix_node: MatrixNode, alpha, beta, depth):
        value = Constant.NEG_INF
        for child_mov in matrix_node.get_child_moves():

            # print(colors.CYAN, "\n max moves", matrix_node.playing, child_mov, colors.ENDC)

            child = get_next_node(matrix_node, child_mov)
            value = max(simple_ab_search(child, alpha, beta, depth-1), value)
            if value >= beta: return value
            alpha = max(value, alpha)
        return value
    
    def get_min(matrix_node: MatrixNode, alpha, beta, depth, cur_best_mov = None):
        value = Constant.POS_INF
        for child_mov in matrix_node.get_child_moves():

            # print(colors.YELLO, "\n min moves", matrix_node.playing, child_mov, colors.ENDC)

            child = get_next_node(matrix_node, child_mov)
            new_val = simple_ab_search(child, alpha, beta, depth-1); value = min(new_val, value)
            if value <= alpha: return value
            beta = min(value, beta)
        return value
    
    if depth == 0 or matrix_node.is_terminate(): return matrix_node.evaluate()

    if TRANS_TABLE.get(matrix_node.hash_str) and (TRANS_TABLE[matrix_node.hash_str].s_val)['depth'] >= depth:
        return (TRANS_TABLE[matrix_node.hash_str].s_val)['value']

    if matrix_node.playing == Constant.WOLF_MAX: value = get_max(matrix_node, alpha, beta, depth)
    elif matrix_node.playing == Constant.SHEEP_MIN: value = get_min(matrix_node, alpha, beta, depth)
    TRANS_TABLE[matrix_node.hash_str].s_val = {'value': value, 'depth': depth}
    return value

@timing
def iter_deep_ab_search(matrix_node: MatrixNode) -> Move:
    best_move = None
    for cur_depth in range(Constant.MIN_DEPTH, Constant.MAX_DEPTH+1, 2):
        value = Constant.NEG_INF if matrix_node.playing == Constant.WOLF_MAX else Constant.POS_INF
        for child_mov in matrix_node.get_child_moves():
            child = get_next_node(matrix_node, child_mov)
            cur_val = simple_ab_search(child, Constant.NEG_INF, Constant.POS_INF, cur_depth)

            if (matrix_node.playing == Constant.WOLF_MAX and cur_val > value
                or matrix_node.playing == Constant.SHEEP_MIN and cur_val < value):
                value, best_move = cur_val, child_mov

    return best_move


def next_move_main(matrix, movemade):
    playing = Constant.WOLF_MAX if movemade else Constant.SHEEP_MIN
    matrix = np.array(matrix)
    matrix_node = MatrixNode(matrix, playing)
    best_move = iter_deep_ab_search(matrix_node)

    # asyncio.run(tidy_table(np.count_nonzero(matrix == Constant.SHEEP_MIN)))

    return best_move.tolist()


def AIAlgorithm(filename, movemade):
    """
    filename: eg. round0/state_59.txt 
    movemade: if True, wolf turn; otherwise, sheep turn
    """
    matrix = np.genfromtxt(filename, delimiter=',')

    print(colors.BLUE, "\n matrix", filename, "\n", matrix, colors.ENDC)
    return next_move_main(matrix, movemade)


if __name__ == '__main__':
    filename = './round0/state_0.txt'
    movemade = False
    AIAlgorithm(filename, movemade)
    print(len(TRANS_TABLE))
