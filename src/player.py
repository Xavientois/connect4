class Player:
    def __init__(self, id, strategy):
        self.id = id
        self.strategy = strategy
        self.score = 0

    def move(self, board_view):
        return self.strategy.move(board_view)

    def update_with_result(self, result):
    	self.score += 1 if result > 0 else 0
    	self.strategy.update(result)