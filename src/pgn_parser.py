"""
PGN Parser for Chess Games
Extracts moves and metadata from PGN format chess games.

This parser does not require the chess library - it parses PGN files directly
using regular expressions.
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ChessGame:
    """Represents a parsed chess game."""
    white: str
    black: str
    result: str
    opening: str
    eco: str  # Encyclopedia of Chess Openings code
    moves: List[str]  # List of moves in SAN notation
    white_moves: List[str]  # Only white's moves
    black_moves: List[str]  # Only black's moves
    first_move: str  # First move (e.g., 'e4', 'd4')
    move_pairs: List[Tuple[str, str]]  # Pairs of (white_move, black_move)


class PGNParser:
    """Parses PGN files and extracts chess moves without external dependencies."""

    # Regex patterns for PGN parsing
    HEADER_PATTERN = re.compile(r'\[(\w+)\s+"([^"]*)"\]')
    MOVE_NUMBER_PATTERN = re.compile(r'\d+\.')
    RESULT_PATTERN = re.compile(r'(1-0|0-1|1/2-1/2|\*)')
    COMMENT_PATTERN = re.compile(r'\{[^}]*\}')
    VARIATION_PATTERN = re.compile(r'\([^)]*\)')
    NAG_PATTERN = re.compile(r'\$\d+')
    CLOCK_PATTERN = re.compile(r'\[%clk[^\]]*\]')
    EVAL_PATTERN = re.compile(r'\[%eval[^\]]*\]')

    # Valid chess move pattern (simplified)
    MOVE_PATTERN = re.compile(
        r'^([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](=[QRBN])?|O-O-O|O-O)[\+#]?$'
    )

    # Common opening classifications based on first moves
    OPENING_CATEGORIES = {
        'e4': 'Kings Pawn',
        'd4': 'Queens Pawn',
        'c4': 'English',
        'Nf3': 'Reti',
        'g3': 'Kings Fianchetto',
        'b3': 'Larsen',
        'f4': 'Birds',
    }

    def __init__(self):
        self.games: List[ChessGame] = []

    def parse_file(self, pgn_path: str) -> List[ChessGame]:
        """
        Parse a PGN file and extract all games.

        Args:
            pgn_path: Path to the PGN file

        Returns:
            List of ChessGame objects
        """
        print(f"Parsing PGN file: {pgn_path}")
        games = []

        with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Split into individual games
        # Games are separated by blank lines followed by headers
        game_blocks = self._split_games(content)

        for block in game_blocks:
            try:
                parsed = self._parse_game_block(block)
                if parsed and len(parsed.moves) >= 4:  # At least 2 full moves
                    games.append(parsed)
            except Exception as e:
                # Skip problematic games
                continue

        self.games = games
        print(f"Successfully parsed {len(games)} games")
        return games

    def _split_games(self, content: str) -> List[str]:
        """Split PGN content into individual game blocks."""
        games = []
        current_game = []
        in_game = False

        for line in content.split('\n'):
            line = line.strip()

            if line.startswith('[') and line.endswith(']'):
                # This is a header line
                if in_game and not current_game[-1].startswith('['):
                    # We were in moves, now starting new game
                    games.append('\n'.join(current_game))
                    current_game = []
                in_game = True
                current_game.append(line)
            elif line:
                # Non-empty, non-header line (probably moves)
                current_game.append(line)
            elif in_game and current_game:
                # Empty line after some content
                if any(self.RESULT_PATTERN.search(l) for l in current_game):
                    # Game is complete
                    games.append('\n'.join(current_game))
                    current_game = []
                    in_game = False

        # Don't forget the last game
        if current_game:
            games.append('\n'.join(current_game))

        return games

    def _parse_game_block(self, block: str) -> Optional[ChessGame]:
        """Parse a single game block."""
        lines = block.split('\n')

        # Extract headers
        headers = {}
        move_lines = []

        for line in lines:
            line = line.strip()
            header_match = self.HEADER_PATTERN.match(line)
            if header_match:
                key, value = header_match.groups()
                headers[key] = value
            elif line and not line.startswith('['):
                move_lines.append(line)

        # Join move lines and extract moves
        move_text = ' '.join(move_lines)
        moves = self._extract_moves(move_text)

        if not moves:
            return None

        # Separate white and black moves
        white_moves = moves[0::2]  # Even indices (0, 2, 4, ...)
        black_moves = moves[1::2]  # Odd indices (1, 3, 5, ...)

        # Create move pairs
        move_pairs = list(zip(white_moves, black_moves))

        # Get opening info
        opening = headers.get('Opening', headers.get('ECO', 'Unknown'))
        eco = headers.get('ECO', '')

        # Determine first move category
        first_move = moves[0] if moves else ''

        return ChessGame(
            white=headers.get('White', 'Unknown'),
            black=headers.get('Black', 'Unknown'),
            result=headers.get('Result', '*'),
            opening=opening,
            eco=eco,
            moves=moves,
            white_moves=white_moves,
            black_moves=black_moves,
            first_move=first_move,
            move_pairs=move_pairs
        )

    def _extract_moves(self, move_text: str) -> List[str]:
        """Extract chess moves from move text."""
        # Remove comments, variations, and annotations
        text = self.COMMENT_PATTERN.sub('', move_text)
        text = self.VARIATION_PATTERN.sub('', text)
        text = self.NAG_PATTERN.sub('', text)
        text = self.CLOCK_PATTERN.sub('', text)
        text = self.EVAL_PATTERN.sub('', text)

        # Remove result
        text = self.RESULT_PATTERN.sub('', text)

        # Remove move numbers (1. 2. 1... etc.)
        text = re.sub(r'\d+\.+', ' ', text)

        # Split by whitespace
        tokens = text.split()

        # Filter valid moves
        moves = []
        for token in tokens:
            token = token.strip()
            # Clean up the token
            token = token.replace('!', '').replace('?', '')

            # Check if it looks like a valid chess move
            if self._is_valid_move(token):
                moves.append(token)

        return moves

    def _is_valid_move(self, token: str) -> bool:
        """Check if a token looks like a valid chess move."""
        if not token or len(token) < 2 or len(token) > 7:
            return False

        # Quick check for common patterns
        # Pawn moves: e4, d5, exd5, e8=Q
        # Piece moves: Nf3, Bb5, Qxd4, Rad1, R1a3
        # Castling: O-O, O-O-O

        if token in ['O-O', 'O-O-O', '0-0', '0-0-0']:
            return True

        # Normalize castling
        if token in ['0-0', '0-0-0']:
            return True

        # Must contain at least one square (a-h followed by 1-8)
        if not re.search(r'[a-h][1-8]', token):
            return False

        # Should not contain invalid characters
        if re.search(r'[^KQRBNa-h1-8x=+#]', token):
            return False

        return True

    def parse_pgn_string(self, pgn_string: str) -> List[ChessGame]:
        """Parse PGN from a string."""
        games = []
        game_blocks = self._split_games(pgn_string)

        for block in game_blocks:
            try:
                parsed = self._parse_game_block(block)
                if parsed and len(parsed.moves) >= 4:
                    games.append(parsed)
            except Exception:
                continue

        self.games.extend(games)
        return games

    def get_games_by_opening(self, opening_move: str) -> List[ChessGame]:
        """
        Filter games by first move (e.g., 'e4', 'd4').

        Args:
            opening_move: The first move to filter by

        Returns:
            List of games starting with that move
        """
        return [g for g in self.games if g.first_move == opening_move]

    def get_move_sequences(self, num_moves: int = 10) -> List[List[str]]:
        """
        Extract sequences of the first N moves from each game.

        Args:
            num_moves: Number of moves to extract

        Returns:
            List of move sequences
        """
        sequences = []
        for game in self.games:
            if len(game.moves) >= num_moves:
                sequences.append(game.moves[:num_moves])
        return sequences

    def get_statistics(self) -> Dict:
        """Get statistics about parsed games."""
        if not self.games:
            return {}

        # Count openings
        opening_counts = defaultdict(int)
        first_move_counts = defaultdict(int)

        for game in self.games:
            first_move_counts[game.first_move] += 1
            if game.eco:
                opening_counts[game.eco[:2]] += 1  # Group by ECO family

        # Average game length
        avg_moves = sum(len(g.moves) for g in self.games) / len(self.games)

        return {
            'total_games': len(self.games),
            'average_moves': avg_moves,
            'first_move_distribution': dict(first_move_counts),
            'eco_distribution': dict(opening_counts),
            'e4_games': first_move_counts.get('e4', 0),
            'd4_games': first_move_counts.get('d4', 0),
            'other_games': sum(v for k, v in first_move_counts.items() if k not in ['e4', 'd4'])
        }

    def extract_all_transitions(self) -> List[Tuple[str, str]]:
        """
        Extract all move transitions (current_move -> next_move) from all games.

        Returns:
            List of (current_move, next_move) tuples
        """
        transitions = []
        for game in self.games:
            for i in range(len(game.moves) - 1):
                transitions.append((game.moves[i], game.moves[i + 1]))
        return transitions

    def extract_white_transitions(self) -> List[Tuple[str, str]]:
        """Extract transitions for white's moves only."""
        transitions = []
        for game in self.games:
            for i in range(len(game.white_moves) - 1):
                transitions.append((game.white_moves[i], game.white_moves[i + 1]))
        return transitions

    def extract_position_transitions(self, position: int = 1) -> List[Tuple[str, str]]:
        """
        Extract transitions at a specific move position.

        Args:
            position: Move number (1-indexed)

        Returns:
            List of (move_at_position, next_move) tuples
        """
        transitions = []
        idx = position - 1  # Convert to 0-indexed

        for game in self.games:
            if len(game.moves) > idx + 1:
                transitions.append((game.moves[idx], game.moves[idx + 1]))

        return transitions


def parse_chess_games(pgn_path: str) -> Tuple[List[ChessGame], Dict]:
    """
    Main function to parse chess games from a PGN file.

    Args:
        pgn_path: Path to PGN file

    Returns:
        Tuple of (list of games, statistics dict)
    """
    parser = PGNParser()
    games = parser.parse_file(pgn_path)
    stats = parser.get_statistics()

    return games, stats


if __name__ == "__main__":
    # Test with a sample file
    import sys
    if len(sys.argv) > 1:
        games, stats = parse_chess_games(sys.argv[1])
        print(f"\nStatistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
