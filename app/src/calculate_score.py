# from mahjong.hand_calculating.hand import HandCalculator
# from mahjong.tile import TilesConverter
# from mahjong.hand_calculating.hand_config import HandConfig
# from mahjong.meld import Meld
#
#
# def calculate_score(
#         tiles: list[str], win_tile_index: int,
#         is_tsumo: bool = False,
#         is_dealer: bool = False,
# ):
#     tiles_136, winning_tile = convert_tiles_to_136_array(tiles, win_tile_index)
#
#     config = HandConfig()
#     config.is_tsumo = is_tsumo
#     config.is_dealer = is_dealer
#
#     calculator = HandCalculator()
#     result = calculator.estimate_hand_value(tiles_136, winning_tile, config=config)
#     print(result.han, result.fu)
#     print(result.cost["main"])
#     print(result.cost["additional"])
#     print(result.yaku)
#     print(result.fu_details)
#
#
# def convert_tiles_to_136_array(tiles: list[str], win_tile_index: int) -> tuple[list, int]:
#     tiles_dict = {"man": "", "pin": "", "sou": "", "honors": ""}
#     letter_to_name = {"m": "man", "p": "pin", "s": "sou", "z": "honors"}
#     for i, tile in enumerate(tiles):
#         tiles_dict[letter_to_name[tile[1]]] += tile[0]
#         if i == win_tile_index:
#             winning_tile = {letter_to_name[tile[1]]: tile[0]}
#     tiles_136 = TilesConverter.string_to_136_array(**tiles_dict)
#     winning_tile = TilesConverter.string_to_136_array(**winning_tile)[0]
#     return tiles_136, winning_tile
#
#
# if __name__ == "__main__":
#     calculate_score(
#         ["2m", "2m", "4m", "4m", "4m", "3p", "3p", "3p", "5p", "6p", "7p", "4s", "4s", "4s"], 13,
#         is_dealer=True, is_tsumo=True,
#     )
