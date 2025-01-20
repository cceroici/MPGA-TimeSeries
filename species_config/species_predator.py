from src.species import Species

predator = Species()
predator.name = "predator"
predator.input_names = ["pos-x", "pos-y", "dist-nr", "prey-detect", "prey-dir-x", "prey-dir-y", "CLK", "field", "dfield-x",
                        "dfield-y", "age", "health",
                        "pher", "pher-dx", "pher-dy"]
predator.output_names = ["M-up", "M-dwn", "M-rght", "M-lft", "M-rnd", "CLK-P+", "CLK-P-", "pher-rel", "MVSPD+",
                                "MVSPD-"]
# inputs idx list:
predator.in_pos_x = 0
predator.in_pos_y = 1
predator.in_dist_nr = 2
predator.in_prey_detect = 3
predator.in_prey_dir_x = 4
predator.in_prey_dir_y = 5
predator.in_clk = 6
predator.in_field = 7
predator.in_field_dx = 8
predator.in_field_dy = 9
predator.in_age = 10
predator.in_health = 11
predator.in_pher = 12
predator.in_pher_dx = 13
predator.in_pher_dy = 14

# output node idx list:
predator.out_mov_up = 0
predator.out_mov_dwn = 1
predator.out_mov_rght = 2
predator.out_mov_lft = 3
predator.out_mov_rnd = 4
predator.out_clk_lim_plus = 5
predator.out_clk_lim_minus = 6
predator.out_pher = 7
predator.out_movspd_plus = 8
predator.out_movspd_minus = 9

predator.Initialize()


