class CatEvalFn():
    """Evaluation function that outputs a score equal to 
    the number of valid moves for the cat."""
    def score_moves(self, game, maximizing_player_turn=True):
        cat_moves=game.valid_moves()
        return len(cat_moves)*2 if maximizing_player_turn else len(cat_moves)-1


    def score_challenge(self, game, maximizing_player_turn=True):
        """Here we have just the score_moves as a value system."""
        return self.score_moves(game) if maximizing_player_turn else -1