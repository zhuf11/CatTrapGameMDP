class CatEvalFn():
    """Evaluation function that outputs a score equal to 
    the number of valid moves for the cat."""

    def score_challenge(self, game, maximizing_player_turn=True):
        """Here we have just proximity as a value system."""
        return self.score_proximity(game) if maximizing_player_turn else -1


    """Evaluation function that outputs a
    score sensitive to how close the cat is to a border."""
    def score_proximity(self, game, maximizing_player_turn=True):
       
        distances=[100,100]
        cat_moves=game.valid_moves()
        #cat_moves=["W","E","SE","SW","NE","NW"]
        for move in cat_moves:
            dist = 0
            i,j = game.cat_i,game.cat_j
            while True:
                dist = dist + 1
                i,j = game.target(i,j,move)
                if (i<0 or i>=game.size or j<0 or j>=game.size):
                    break
                if game.tiles[i][j] != 0:
                    dist = dist*5
                    break
            distances.append(dist)

        distances.sort() 
        return game.size*2-(distances[0] if maximizing_player_turn else distances[1])