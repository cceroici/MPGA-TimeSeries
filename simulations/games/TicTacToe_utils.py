import numpy as np
import cv2


def legal_move(game_state, move_query):
    if move_query == -1:
        return False
    if not game_state[move_query] == 0:
        return False
    return True


def check_winner(game_state):
    # Possible winning combinations of indices
    winning_combinations = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]  # Diagonals
    ]

    for combo in winning_combinations:
        if game_state[combo[0]] == game_state[combo[1]] == game_state[combo[2]] and game_state[combo[0]] != 0:
            return game_state[combo[0]]

    if 0 not in game_state:
        return 0  # It's a draw
    else:
        return None  # No winner yet


def update_game_state(game_state, move=None, is_player=True):
    if move is None:
        move = get_ai_move(board_state=game_state, mode="best")
    if move is not None:
        game_state[move] = 1 if is_player else -1

    return game_state


def get_ai_move(board_state, mode="first-empty"):
    if mode == "first-empty":
        for i in range(len(board_state)):
            if board_state[i] == 0:
                return i
    elif mode == "random":
        empty = get_empty_indices(board_state)
        if len(empty) == 0:
            return None
        return np.random.choice(empty)
    elif mode == "best":
        return find_best_move(board=board_state, player_symbol=-1)


def get_designer_genes(gene_count, output_offset):
    from src.Genome import Encode_Connection_Gene, latent_gene

    genes = []
    genes.append(Encode_Connection_Gene(src_node=18, dst_node=6 + output_offset, weight=-1, bias=1))
    genes.append(Encode_Connection_Gene(src_node=18, dst_node=7 + output_offset, weight=1, bias=-0.5))
    genes.append(Encode_Connection_Gene(src_node=19, dst_node=8 + output_offset, weight=1, bias=-0.5))
    genes.append(Encode_Connection_Gene(src_node=19, dst_node=7 + output_offset, weight=-1, bias=0))
    for i in range(gene_count - len(genes)):
        genes.append(latent_gene())

    return genes


def get_empty_indices(game_state):
    return [i for i, cell in enumerate(game_state) if cell == 0]


def evaluate_board(board):
    # Define all possible winning combinations
    winning_combinations = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]             # Diagonals
    ]

    for combo in winning_combinations:
        if board[combo[0]] == board[combo[1]] == board[combo[2]] != 0:
            return board[combo[0]]

    if 0 not in board:
        return 0  # Draw
    return None  # Game still ongoing


def minimax(board, depth, maximizing_player, player_symbol, opponent_symbol):
    if maximizing_player:
        symbol = player_symbol
    else:
        symbol = opponent_symbol

    result = evaluate_board(board)
    if result is not None:
        if result == player_symbol:
            return 10 - depth
        elif result == opponent_symbol:
            return depth - 10
        return 0

    if maximizing_player:
        max_eval = -float('inf')
        for i in range(9):
            if board[i] == 0:
                board[i] = symbol
                eval = minimax(board, depth + 1, False, player_symbol, opponent_symbol)
                board[i] = 0
                max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for i in range(9):
            if board[i] == 0:
                board[i] = symbol
                eval = minimax(board, depth + 1, True, player_symbol, opponent_symbol)
                board[i] = 0
                min_eval = min(min_eval, eval)
        return min_eval

def find_best_move(board, player_symbol):
    opponent_symbol = -player_symbol
    best_move = -1
    best_score = -float('inf')

    for i in range(9):
        if board[i] == 0:
            board[i] = player_symbol
            move_score = minimax(board, 0, False, player_symbol, opponent_symbol)
            board[i] = 0

            if move_score > best_score:
                best_score = move_score
                best_move = i

    return best_move


class TicTacToe_animation:
    def __init__(self):
        self.board = np.zeros(9, dtype=int)
        self.probabilities = None
        self.frame_size = (400, 400)  # Adjust the frame size as needed
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.text_scale = 2
        self.text_thickness = 3
        self.board_states = []
        self.probabilities = []

    def update_board(self, new_board):
        self.board = new_board

    def update_state(self, state, probabilities=None):
        self.board_states.append(state)
        if probabilities is not None:
            probabilities = np.array(probabilities)
            probabilities /= (probabilities.max() + 1e-10)
            self.probabilities.append(probabilities)

    def generate_frame(self, t, probabilities=None):
        frame = np.ones((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8) * 255
        if check_winner(self.board) == -1:
            frame[:, :, 1] = 50
            frame[:, :, 2] = 50
        elif check_winner(self.board) == 1:
            frame[:, :, 0] = 50
            frame[:, :, 2] = 50
        elif check_winner(self.board) == 1:
            frame[:, :, 0] = 128
            frame[:, :, 1] = 128
            frame[:, :, 2] = 128

        cell_width = self.frame_size[1] // 3
        cell_height = self.frame_size[0] // 3

        # Draw vertical lines
        for i in range(1, 3):
            cv2.line(frame, (i * cell_width, 0), (i * cell_width, self.frame_size[0]), (0, 0, 0), 2)

        # Draw horizontal lines
        for i in range(1, 3):
            cv2.line(frame, (0, i * cell_height), (self.frame_size[1], i * cell_height), (0, 0, 0), 2)

        for i in range(3):
            for j in range(3):
                index = i * 3 + j
                cell_center = ((j + 0.5) * cell_width, (i + 0.5) * cell_height)
                if self.board[index] == 1:
                    marker = "X"
                    marker_color = (0, 0, 100)  # Red color
                elif self.board[index] == -1:
                    marker = "O"
                    marker_color = (255, 0, 0)  # Blue color
                else:
                    marker = ""
                    marker_color = (0, 0, 0)  # Black color
                    if probabilities is not None:
                        probability = probabilities[index]
                        faded_red_opacity = 0.5 * probability
                        marker_color = (0, 0, 255)  # Faded red color
                        frame = self.apply_opacity(frame, marker_color, faded_red_opacity, cell_center, cell_width,
                                                   cell_height)
                cv2.putText(frame, marker, (int(cell_center[0]), int(cell_center[1])), self.font, self.text_scale,
                            marker_color, self.text_thickness)

        # header_text = f"Time Step: {t}"
        # frame = cv2.copyMakeBorder(frame, 100, 0, 0, 0, cv2.BORDER_CONSTANT)
        # cv2.putText(frame, header_text, (10, 40), self.font, self.text_scale, (0, 0, 0), self.text_thickness)

        return frame

    def apply_opacity(self, frame, color, opacity, cell_center, cell_width, cell_height):
        overlay = frame.copy()
        cv2.putText(overlay, "X", (int(cell_center[0]), int(cell_center[1])), self.font, self.text_scale, color,
                    self.text_thickness)
        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
        return frame

    def generate_animation(self, output_path):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        out = cv2.VideoWriter(output_path, fourcc, 1, self.frame_size)

        for time_step, (state, probabilities) in enumerate(
                zip(self.board_states, self.probabilities or [None] * len(self.board_states))):

            if time_step == 0:
                self.update_board(np.zeros(len(state)))
                frame = self.generate_frame(time_step)
                out.write(frame)
                frame = self.generate_frame(time_step)
                out.write(frame)

            self.update_board(state)
            frame = self.generate_frame(time_step, probabilities)
            out.write(frame)
            if time_step == len(self.board_states)-1:
                frame = self.generate_frame(time_step, probabilities)
                out.write(frame)
                frame = self.generate_frame(time_step, probabilities)
                out.write(frame)


        out.release()


if __name__ == "__main__":

    # Example usage
    game = TicTacToe_animation()

    # Simulate game progress (example states)
    board_state = np.zeros(9)
    states = []
    for i in range(6):
        states.append(board_state.copy())
        board_state = update_game_state(board_state, is_player=True)
        board_state = update_game_state(board_state, is_player=False)
        game.update_state(state=np.array(board_state).copy(), probabilities=np.random.rand(9))

    game.generate_animation('tictactoe_animation.mp4')
