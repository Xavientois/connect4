class Board:
    COLUMN_COUNT = 7
    ROW_COUNT = 6
    LINE_LENGTH = 4
    EMPTY_CELL = '.'

    def __init__(self):
        self.cols = Board.COLUMN_COUNT
        self.rows = Board.ROW_COUNT
        self.line_length = Board.LINE_LENGTH
        self.board = [[Board.EMPTY_CELL for c in range(self.cols)] for r in range(self.rows)]

    def drop(self, col, value):
        if col < 0 or col >= self.cols:
            raise ValueError('Bad column value {}'.format(col))

        col_values = list([r[col] for r in self.board])
        index_of_highest_empty_cell = len(self.board)
        while True:
            if col_values[index_of_highest_empty_cell - 1] == Board.EMPTY_CELL:
                index_of_highest_empty_cell -= 1
                if index_of_highest_empty_cell == 0:
                    break
            else:
                break

        if index_of_highest_empty_cell < len(self.board):
            self._set(col, index_of_highest_empty_cell, value)
            return self._find_lines(col, index_of_highest_empty_cell)

        else:
            raise ValueError('column {} is full'.format(col))

    def get_valid_moves(self):
        return [i for i, v in enumerate(self.board[self.rows-1]) if v == Board.EMPTY_CELL]

    def clone(self):
        board = Board()
        board.cols = self.cols
        board.rows = self.rows
        board.line_length = self.line_length
        board.board = [row[:] for row in self.board]
        return board

    def _set(self, col, row, value):
        self._check_coords(col, row)
        self.board[row][col] = value

    def _get(self, col, row):
        self._check_coords(col, row)
        return self.board[row][col]

    def _check_coords(self, col, row):
        if col < 0 or col >= self.cols or row < 0 or row >= self.rows:
            raise ValueError('Bad coords col={} row={}'.format(col, row))

    def _find_lines(self, col, row):
        u_count = self._count_in_direction(col, row, -1, 0)
        d_count = self._count_in_direction(col, row, 1, 0)
        l_count = self._count_in_direction(col, row, 0, -1)
        r_count = self._count_in_direction(col, row, 0, 1)
        ur_count = self._count_in_direction(col, row, -1, 1)
        ul_count = self._count_in_direction(col, row, -1, -1)
        dr_count = self._count_in_direction(col, row, 1, 1)
        dl_count = self._count_in_direction(col, row, 1, -1)

        return list([l for l in [u_count+d_count-1, r_count+l_count-1, ur_count+dl_count-1, ul_count+dr_count-1] if l >= self.line_length])

    def _count_in_direction(self, init_col, init_row, row_delta, col_delta):
        col = init_col
        row = init_row
        count = 0
        value = self._get(col, row)

        try:
            while self._get(col, row) == value:
                count += 1
                col += col_delta
                row += row_delta

        finally:
            return count

    def __str__(self):
        return '\n'.join([' '.join(map(str,r)) for r in reversed(self.board)])
