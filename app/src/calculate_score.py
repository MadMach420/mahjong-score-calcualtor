from mahjong.hand_calculating.hand import HandCalculator
from mahjong.tile import TilesConverter
from mahjong.hand_calculating.hand_config import HandConfig
from mahjong.meld import Meld


def calculate_score(tiles: list[str], win_tile_index: int):
    tiles_136, winning_tile = convert_tiles_to_136_array(tiles)
    print(tiles_136)
    print(winning_tile)



def convert_tiles_to_136_array(tiles: list[str], win_tile_index: int = None) -> tuple[list, list]:
    tiles_dict = {"man": "", "pin": "", "sou": "", "honors": ""}
    letter_to_name = {"m": "man", "p": "pin", "s": "sou", "z": "honors"}
    for i, tile in enumerate(tiles):
        tiles_dict[letter_to_name[tile[1]]] += tile[0]
        if i == win_tile_index:
            winning_tile = {letter_to_name[tile[1]]: tile[0]}
    tiles_136 = TilesConverter.to_136_array(**tiles_dict)
    winning_tile = TilesConverter.to_136_array(**winning_tile)
    return tiles_136, winning_tile


# if __name__ == "__main__":
