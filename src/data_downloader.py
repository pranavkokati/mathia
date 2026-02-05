"""
Data Downloader for Lichess Elite Grandmaster Games
Downloads real chess games in PGN format from Lichess database.
"""

import requests
import os
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm


class LichessDownloader:
    """Downloads chess games from Lichess database."""

    # Lichess elite database URL (smaller, curated grandmaster games)
    # Using standard rated games database which is more accessible
    LICHESS_ELITE_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst"

    # Alternative: Use Lichess API to get games from top players
    LICHESS_API_BASE = "https://lichess.org/api"

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_games_from_player(self, username: str, max_games: int = 500) -> str:
        """
        Download games from a specific Lichess player.

        Args:
            username: Lichess username
            max_games: Maximum number of games to download

        Returns:
            Path to the downloaded PGN file
        """
        url = f"{self.LICHESS_API_BASE}/games/user/{username}"
        params = {
            "max": max_games,
            "rated": "true",
            "perfType": "classical,rapid,blitz",
            "opening": "true"
        }
        headers = {
            "Accept": "application/x-chess-pgn"
        }

        output_file = self.data_dir / f"{username}_games.pgn"

        print(f"Downloading games from {username}...")
        response = requests.get(url, params=params, headers=headers, stream=True)

        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded to {output_file}")
            return str(output_file)
        else:
            raise Exception(f"Failed to download: {response.status_code}")

    def download_grandmaster_games(self, num_games: int = 2000) -> str:
        """
        Download games from multiple grandmaster-level players.

        Args:
            num_games: Approximate total number of games to download

        Returns:
            Path to the combined PGN file
        """
        # Top Lichess players (titled players with GM/IM ratings)
        top_players = [
            "DrNykterstein",  # Magnus Carlsen
            "Hikaru",         # Hikaru Nakamura
            "GMWSO",          # Wesley So
            "FairChess_on_YouTube",  # Titled player
            "Zhigalko_Sergei",  # GM Sergei Zhigalko
            "nihalsarin",     # Nihal Sarin
            "Polish_fighter3",  # Titled player
            "Msb2",           # Mikhail Botvinnik memorial account
            "Fins",           # John Bartholomew
            "penguingm1",     # Andrew Tang
        ]

        games_per_player = max(100, num_games // len(top_players))
        all_games = []

        combined_file = self.data_dir / "grandmaster_games.pgn"

        with open(combined_file, 'w') as outfile:
            for player in tqdm(top_players, desc="Downloading from players"):
                try:
                    pgn_content = self._fetch_player_games(player, games_per_player)
                    if pgn_content:
                        outfile.write(pgn_content)
                        outfile.write("\n\n")
                except Exception as e:
                    print(f"  Warning: Could not download from {player}: {e}")
                    continue

        print(f"\nCombined games saved to {combined_file}")
        return str(combined_file)

    def _fetch_player_games(self, username: str, max_games: int) -> str:
        """Fetch games from a player and return PGN content."""
        url = f"{self.LICHESS_API_BASE}/games/user/{username}"
        params = {
            "max": max_games,
            "rated": "true",
            "opening": "true"
        }
        headers = {
            "Accept": "application/x-chess-pgn"
        }

        response = requests.get(url, params=params, headers=headers, timeout=60)

        if response.status_code == 200:
            return response.text
        elif response.status_code == 429:
            print(f"  Rate limited, waiting...")
            import time
            time.sleep(60)
            return self._fetch_player_games(username, max_games)
        else:
            return ""

    def download_sample_games(self) -> str:
        """
        Download a curated sample of games from the Lichess open database.
        Uses the Lichess studies API to get annotated master games.

        Returns:
            Path to the downloaded PGN file
        """
        # Download from Lichess opening explorer (top games)
        output_file = self.data_dir / "sample_games.pgn"

        # We'll use the Lichess masters database API
        # This gives us historical master games
        openings = [
            ("e2e4", "e7e5"),  # King's Pawn
            ("e2e4", "c7c5"),  # Sicilian
            ("d2d4", "d7d5"),  # Queen's Pawn
            ("d2d4", "g8f6"),  # Indian Defense
            ("c2c4", "e7e5"),  # English
        ]

        all_games = []

        print("Fetching master games from opening explorer...")
        for white_move, black_move in tqdm(openings):
            try:
                games = self._fetch_from_opening_explorer(white_move, black_move)
                all_games.extend(games)
            except Exception as e:
                print(f"  Error fetching {white_move}/{black_move}: {e}")

        # Write to PGN file
        with open(output_file, 'w') as f:
            for game in all_games:
                f.write(game + "\n\n")

        print(f"Saved {len(all_games)} games to {output_file}")
        return str(output_file)

    def _fetch_from_opening_explorer(self, white_move: str, black_move: str) -> list:
        """Fetch top games from the opening explorer."""
        url = "https://explorer.lichess.ovh/masters"
        params = {
            "play": f"{white_move},{black_move}",
            "topGames": 50,  # Get top 50 games per opening
        }

        response = requests.get(url, params=params, timeout=30)

        if response.status_code == 200:
            data = response.json()
            games = []

            # Fetch the actual game PGNs
            for game_info in data.get("topGames", []):
                game_id = game_info.get("id")
                if game_id:
                    pgn = self._fetch_masters_game(game_id)
                    if pgn:
                        games.append(pgn)

            return games
        return []

    def _fetch_masters_game(self, game_id: str) -> str:
        """Fetch a specific master game by ID."""
        url = f"https://explorer.lichess.ovh/masters/pgn/{game_id}"

        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                return response.text
        except:
            pass
        return ""


def download_chess_data(num_games: int = 2000, data_dir: str = "data") -> str:
    """
    Main function to download chess games.

    Args:
        num_games: Number of games to download
        data_dir: Directory to save data

    Returns:
        Path to the PGN file
    """
    downloader = LichessDownloader(data_dir)

    # First try to download from grandmaster players
    try:
        return downloader.download_grandmaster_games(num_games)
    except Exception as e:
        print(f"Could not download from players: {e}")
        print("Trying alternative source...")
        return downloader.download_sample_games()


if __name__ == "__main__":
    pgn_file = download_chess_data(num_games=2000)
    print(f"Downloaded games to: {pgn_file}")
