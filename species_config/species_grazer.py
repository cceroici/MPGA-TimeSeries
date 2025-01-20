from src.species import Species

grazer = Species()
grazer.name = "grazer"
grazer.input_names = ["pos-x", "pos-y", "dist-nr", "pred-detect", "pred-dir-x", "pred-dir-y", "CLK", "field", "dfield-x", "dfield-y", "age", "health",
                               "pher", "pher-dx", "pher-dy", "pop-dens", "pop-dens-dx", "pop-dens-dy"]
grazer.output_names = ["M-up", "M-dwn", "M-rght", "M-lft", "M-rnd", "CLK-P+", "CLK-P-", "pher-rel", "MVSPD+",
                                "MVSPD-"]

# inputs idx list:
grazer.in_pos_x = 0
grazer.in_pos_y = 1
grazer.in_dist_nr = 2
grazer.in_pred_detect = 3
grazer.in_pred_dir_x = 4
grazer.in_pred_dir_y = 5
grazer.in_clk = 6
grazer.in_field = 7
grazer.in_field_dx = 8
grazer.in_field_dy = 9
grazer.in_age = 10
grazer.in_health = 11
grazer.in_pher = 12
grazer.in_pher_dx = 13
grazer.in_pher_dy = 14
grazer.in_PD = 15
grazer.in_PD_dx = 16
grazer.in_PD_dy = 17

# output node idx list:
grazer.out_mov_up = 0
grazer.out_mov_dwn = 1
grazer.out_mov_rght = 2
grazer.out_mov_lft = 3
grazer.out_mov_rnd = 4
grazer.out_clk_lim_plus = 5
grazer.out_clk_lim_minus = 6
grazer.out_pher = 7
grazer.out_movspd_plus = 8
grazer.out_movspd_minus = 9

print("WARNING: USING GRAZER SPECIES")
grazer.Initialize(0)