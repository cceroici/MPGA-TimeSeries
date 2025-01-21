import shutil
import numpy as np
from tqdm import tqdm
import cv2
import pickle
import matplotlib as mpl
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm, animation
from scipy.signal import fftconvolve
from src.genome import genetic_histogram, check_genome_has_connections
from src.fitness import calc_fitness_roi_validation


def ema(input, time_period=10):  # For time period = 10
    t_ = time_period - 1
    ema = np.zeros_like(input, dtype=float)
    multiplier = 2.0 / (time_period + 1)
    # multiplier = 1 - multiplier
    for i in range(len(input)):
        # Special Case
        if i > t_:
            ema[i] = (input[i] - ema[i - 1]) * multiplier + ema[i - 1]
        else:
            ema[i] = np.mean(input[:i + 1])
    return ema


def overlay_transparent(background, overlay, x=0, y=0):
    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype) * 255
            ],
            axis=2,
        )

    # overlay_image = overlay[..., :3]
    overlay_image = overlay
    mask = overlay[..., 3:] / 255.0

    background[y:y + h, x:x + w] = (1.0 - mask) * background[y:y + h, x:x + w] + mask * overlay_image

    return background


def overlay_layers(layers):
    temp = layers[0]
    for l in range(1, len(layers)):
        temp = overlay_transparent(temp, layers[l])
    return temp


def overlay_layer_frames(layer_frames, remove_alpha=False):
    no_frames = len(layer_frames[0])
    no_layers = len(layer_frames)
    combined_frames = []
    for f in range(no_frames):
        layers = np.copy(layer_frames[0][f])
        for i in range(1, no_layers):
            layers[layer_frames[i][f][:, :, 3].nonzero()] = layer_frames[i][f][layer_frames[i][f][:, :, 3].nonzero()]
        combined_frames.append(layers)

    return combined_frames


def frames_to_video(frames, output_path, fps=20, convert_bgr=False):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fps=fps, fourcc=fourcc,
                          frameSize=(frames[0].shape[1], frames[0].shape[0]))

    for i in range(len(frames)):
        if convert_bgr:
            out.write(cv2.cvtColor(frames[i], cv2.COLOR_BGRA2RGB))
        else:
            out.write(frames[i])
    out.release()


def display_sim(gen_no, organisms, sim_width, sim_height, resolution, field=None, max_health=None, pher_release=None,
                extra_text=None, show_org_label=False):
    """
    Load data for a single frame and generate a frame of the preview animation
    :param gen_no:
    :param organisms:
    :param sim_width:
    :param sim_height:
    :param resolution:
    :param field:
    :param max_health:
    :param pher_release:
    :param extra_text:
    :param show_org_label:
    :return:
    """
    canvas = np.zeros((resolution[0], resolution[1], 3)).astype('uint8')

    if field is not None:
        cmap_field = LinearSegmentedColormap.from_list('field cmap', [[1, 0.3, 0], [0, 1, 0.3]])
        field_rgb = (cmap_field((np.transpose(field, (1, 0)) - field.min()) / 2) * 255.0).astype('uint8')
        canvas = cv2.resize(field_rgb, resolution)
    else:
        canvas += 255

    cmap_health = LinearSegmentedColormap.from_list('health cmap', [[0, 0, 0], [1, 1, 0], [0, 1, 0], [1, 1, 1]])

    for i, org in enumerate(organisms):
        if org.alive:
            if max_health is not None:
                cvalue = (np.array(cmap_health(org.health / max_health)) * 255.0).astype('uint8')
                fill_color = (int(cvalue[0]), int(cvalue[1]), int(cvalue[2]))
            else:
                fill_color = (0, 200, 255)
            border_color = (0, 0, 0)
            if pher_release is not None:
                if pher_release[i] > 0.9:
                    border_color = (255, 255, 0)
            coord = (int((org.pos[0] + sim_width / 2) / sim_width * resolution[0]),
                     int((org.pos[1] + sim_height / 2) / sim_height * resolution[1]))
            canvas = cv2.circle(canvas, center=np.array(coord), radius=3, color=fill_color, thickness=-1)
            canvas = cv2.circle(canvas, center=np.array(coord), radius=4, color=border_color, thickness=1)
            if show_org_label:
                cv2.putText(canvas, "{}".format(i),
                            coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    np.fliplr(canvas)
    np.flipud(canvas)
    cv2.putText(canvas, "Generation {}".format(gen_no),
                (100, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if extra_text is not None:
        cv2.putText(canvas, extra_text,
                    (100, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return canvas


def display_sim_all_frames(gen_no, pos, alive, health, sim_width, sim_height, resolution, field=None, pher_release=None,
                           max_health=None, pher_sat=None, extra_text=None, show_org_label=False, pheromone_map=None,
                           video_output_path=None, fps=30):
    """
    Load lists containing information across all frames and generate a preview video .mp4
    :param gen_no:
    :param pos:
    :param alive:
    :param health:
    :param sim_width:
    :param sim_height:
    :param resolution:
    :param field:
    :param pher_release:
    :param max_health:
    :param pher_sat:
    :param extra_text:
    :param show_org_label:
    :param pheromone_map:
    :param video_output_path:
    :param fps:
    :return:
    """
    no_frames = len(pos)
    canvas = np.zeros((no_frames, resolution[0], resolution[1], 3)).astype('uint8')

    if field is not None:
        cmap_field = LinearSegmentedColormap.from_list('field cmap', [[1, 0.3, 0], [0, 1, 0.3]])

        for f in range(no_frames):
            field_frame = field[:, :, f]
            field_rgb = (cmap_field((np.transpose(field_frame, (1, 0)) - field_frame.min()) / 2) * 255.0).astype(
                'uint8')
            canvas[f, :, :, :] = cv2.resize(field_rgb[:, :, 0:3], resolution)
    else:
        canvas += 255

    cmap_health = LinearSegmentedColormap.from_list('health cmap', [[0, 0, 0], [1, 1, 0], [0, 1, 0], [1, 1, 1]])

    for f in range(no_frames):
        canvas_temp = canvas[f, :, :, :]
        for i in range(len(pos[f])):
            if alive[f][i]:
                if max_health is not None:
                    cvalue = (np.array(cmap_health(health[f][i] / max_health)) * 255.0).astype('uint8')
                    fill_color = (int(cvalue[0]), int(cvalue[1]), int(cvalue[2]))
                else:
                    fill_color = (0, 200, 255)
                border_color = (0, 0, 0)
                if pher_release is not None:
                    if pher_release[i] > 0.9:
                        border_color = (255, 255, 0)
                coord = (int((pos[f][i][0] + sim_width / 2) / sim_width * resolution[0]),
                         int((pos[f][i][1] + sim_height / 2) / sim_height * resolution[1]))
                canvas_temp = cv2.circle(canvas_temp, center=np.array(coord), radius=3, color=fill_color, thickness=-1)
                canvas_temp = cv2.circle(canvas_temp, center=np.array(coord), radius=4, color=border_color, thickness=1)
                if show_org_label:
                    cv2.putText(canvas_temp, "{}".format(i),
                                coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        np.fliplr(canvas_temp)
        np.flipud(canvas_temp)
        cv2.putText(canvas_temp, "Generation {}".format(gen_no),
                    (100, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if extra_text is not None:
            cv2.putText(canvas_temp, extra_text,
                        (100, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        canvas_temp = cv2.cvtColor(canvas_temp, cv2.COLOR_BGR2RGB)
        canvas[f, :, :, :] = canvas_temp

    if pheromone_map is not None and pher_sat is not None:
        canvas_full = np.zeros((no_frames, resolution[0], resolution[1] * 2, 3)).astype('uint8')
        for f in range(no_frames):
            cmap_pher = LinearSegmentedColormap.from_list('pheromone cmap', [[1, 1, 1], [0, 0, 0]])
            pher_rgb = (cmap_pher((np.transpose(pheromone_map[f, :, :], (1, 0))) / pher_sat) * 255.0).astype(
                'uint8')
            canvas_temp_pher = cv2.resize(pher_rgb[:, :, 0:3], resolution)
            canvas_full[f, :, :, :] = np.hstack((canvas[f, :, :, :], canvas_temp_pher))

    if video_output_path is None:
        return canvas
    else:
        frames_to_video(frames=canvas, output_path=video_output_path, fps=20)


def convolutional_render_points(coords, marker_kernel, marker_edge_kernel, resolution,
                                marker_c, edge_c, alive_mask=None, dead_marker_c=None):
    points = np.zeros(resolution)
    if alive_mask is not None:
        points_dead = np.zeros(resolution)
        coords_alive = coords[alive_mask == 1, :]
        coords_dead = coords[alive_mask == 0, :]
        points[coords_alive[:, 1], coords_alive[:, 0]] = 1
        points_dead[coords_dead[:, 1], coords_dead[:, 0]] = 1
    else:
        points[coords[:, 1], coords[:, 0]] = 1

    cnv = fftconvolve(points, marker_kernel, mode='same')
    cnv[cnv > 1] = 1
    cnv[cnv < 0] = 0
    new_frame = cnv
    new_frame = np.tile(np.expand_dims(new_frame, 2), (1, 1, 4))
    new_frame[:, :, 0] *= marker_c[0]
    new_frame[:, :, 1] *= marker_c[1]
    new_frame[:, :, 2] *= marker_c[2]
    new_frame[:, :, 3] *= 255
    new_frame = new_frame.astype('uint8')

    if marker_edge_kernel is not None:
        cnv = fftconvolve(points, marker_edge_kernel, mode='same')
        cnv[cnv > 1] = 1
        cnv[cnv < 0] = 0
        new_frame_edge = cnv
        new_frame_edge = np.tile(np.expand_dims(new_frame_edge, 2), (1, 1, 4))
        new_frame_edge[:, :, 0] *= edge_c[0]
        new_frame_edge[:, :, 1] *= edge_c[1]
        new_frame_edge[:, :, 2] *= edge_c[2]
        new_frame_edge[:, :, 3] *= 255
        new_frame_edge = new_frame_edge.astype('uint8')
        new_frame[new_frame_edge[:, :, 3].nonzero()] = new_frame_edge[new_frame_edge[:, :, 3].nonzero()]
        # new_frame = overlay_transparent(new_frame, new_frame_edge)

    if alive_mask is not None and points_dead.shape[0] > 0:
        cnv = fftconvolve(points_dead, marker_kernel, mode='same')
        cnv[cnv > 1] = 1
        cnv[cnv < 0] = 0
        dead_frame = cnv
        dead_frame = np.tile(np.expand_dims(dead_frame, 2), (1, 1, 4))
        dead_frame[:, :, 0] *= dead_marker_c[0]
        dead_frame[:, :, 1] *= dead_marker_c[1]
        dead_frame[:, :, 2] *= dead_marker_c[2]
        dead_frame[:, :, 3] *= 255
        dead_frame = dead_frame.astype('uint8')
        new_frame[dead_frame[:, :, 3].nonzero()] = dead_frame[dead_frame[:, :, 3].nonzero()]
        # new_frame = overlay_transparent(new_frame, dead_frame)

    return new_frame


def display_sim_anim_convolution(gen_no, pos, alive, health, sim_width, sim_height, resolution, field=None,
                                 pher_release=None,
                                 max_health=None, pher_sat=None, extra_text=None, pheromone_map=None,
                                 fitness=None,
                                 org_radius=3,
                                 video_output_path=None, fps=30,
                                 highlight_target_org=None):
    no_frames = len(pos)
    frames = []

    p2 = np.zeros((org_radius * 4 + 1, org_radius * 4 + 1)).astype('uint8')
    p2 = cv2.circle(p2, (org_radius * 2, org_radius * 2), org_radius, 1, -1)
    p2 = p2 / p2.max()

    p3 = np.zeros((org_radius * 4 + 1, org_radius * 4 + 1)).astype('uint8')
    p3 = cv2.circle(p3, (org_radius * 2, org_radius * 2), org_radius, 1, 2)
    p3 = p3 / p3.max()

    pos = np.array(pos)
    coords = np.array([((pos[:, :, 0] + sim_width / 2) / sim_width * resolution[0]).astype('int'),
                       ((pos[:, :, 1] + sim_height / 2) / sim_height * resolution[1]).astype('int')])
    coords = np.transpose(coords, (1, 2, 0))

    cmap_field = LinearSegmentedColormap.from_list('field cmap',
                                                   [[1, 0.3, 0], [0.5, 0.5, 0.5], [0, 1, 0.3]])  # field colormap
    cmap_pher = LinearSegmentedColormap.from_list('pheromone cmap', [[1, 1, 1], [0.5, 0, .5]])  # pheromone colormap
    org_fill_color = (200, 225, 200)
    org_edge_color = (0, 0, 0)

    if field is not None:
        field_rgb = (cmap_field(np.transpose(field, (1, 0, 2))) * 255.0).astype(
            'uint8')
        field_rgb_resize = []
        for f in range(no_frames):
            field_rgb_resize.append(cv2.resize(field_rgb[:, :, f, 0:3], resolution))
        field_rgb_resize = np.array(field_rgb_resize)

    if pheromone_map is not None and pher_sat is not None:

        pher_rgb = (cmap_pher(
            (np.transpose(pheromone_map, (2, 1, 0))) / (pheromone_map.max() + 0.0001)) * 255.0).astype(
            'uint8')
        pher_rgb_resize = []
        for f in range(no_frames):
            pher_rgb_resize.append(cv2.resize(pher_rgb[:, :, f, 0:3], resolution))
        pher_rgb_resize = np.array(pher_rgb_resize)

    for f in range(no_frames):
        if field is not None:
            '''            field_frame = field[:, :, f]
                        field_rgb = (cmap_field((np.transpose(field_frame, (1, 0)) - field_frame.min()) / 2) * 255.0).astype(
                            'uint8')'''
            new_frame_canvas = field_rgb_resize[f, :, :, 0:3]
        else:
            new_frame_canvas = (np.ones((resolution[0], resolution[1], 3)) * 255.0).astype('uint8')

        p1 = np.zeros(resolution)
        p1[coords[f, :, 1], coords[f, :, 0]] = 1

        cnv = fftconvolve(p1, p2, mode='same')
        cnv[cnv > 1] = 1
        cnv[cnv < 0] = 0
        # new_frame_org = np.flipud(np.fliplr(cnv))
        new_frame_org = cnv
        new_frame_org = np.tile(np.expand_dims(new_frame_org, 2), (1, 1, 4))
        new_frame_org[:, :, 0] *= org_fill_color[0]
        new_frame_org[:, :, 1] *= org_fill_color[1]
        new_frame_org[:, :, 2] *= org_fill_color[2]
        new_frame_org[:, :, 3] *= 255
        new_frame_org = new_frame_org.astype('uint8')

        if fitness is not None:
            fit_sort_idx = np.argsort(fitness)
            new_frame_org = cv2.circle(new_frame_org, (coords[f, fit_sort_idx[-1], 0], coords[f, fit_sort_idx[-1], 1]),
                                       org_radius, (0, 255, 0, 200), -1)
            new_frame_org = cv2.circle(new_frame_org, (coords[f, fit_sort_idx[-2], 0], coords[f, fit_sort_idx[-2], 1]),
                                       org_radius, (0, 100, 255, 200), -1)
            new_frame_org = cv2.circle(new_frame_org, (coords[f, fit_sort_idx[-3], 0], coords[f, fit_sort_idx[-3], 1]),
                                       org_radius, (255, 0, 0, 200), -1)

        if highlight_target_org is not None:
            new_frame_org = cv2.circle(new_frame_org,
                                       (coords[f, highlight_target_org, 0], coords[f, highlight_target_org, 1]),
                                       org_radius + 3, (0, 0, 0, 255), -1)
            new_frame_org = cv2.circle(new_frame_org,
                                       (coords[f, highlight_target_org, 0], coords[f, highlight_target_org, 1]),
                                       org_radius + 2, (255, 255, 255, 255), -1)
            new_frame_org = cv2.circle(new_frame_org,
                                       (coords[f, highlight_target_org, 0], coords[f, highlight_target_org, 1]),
                                       org_radius, (255, 255, 0, 255), -1)

        cnv = fftconvolve(p1, p3, mode='same')
        cnv[cnv > 1] = 1
        cnv[cnv < 0] = 0
        new_frame_org_edge = cnv
        new_frame_org_edge = np.tile(np.expand_dims(new_frame_org_edge, 2), (1, 1, 4))
        new_frame_org_edge[:, :, 0] *= org_edge_color[0]
        new_frame_org_edge[:, :, 1] *= org_edge_color[1]
        new_frame_org_edge[:, :, 2] *= org_edge_color[2]
        new_frame_org_edge[:, :, 3] *= 255
        new_frame_org_edge = new_frame_org_edge.astype('uint8')

        # composite layers:
        new_frame = overlay_transparent(new_frame_canvas, new_frame_org_edge)
        new_frame = overlay_transparent(new_frame, new_frame_org)
        # Title
        cv2.putText(new_frame, "Generation {}".format(gen_no),
                    (100, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if extra_text is not None:
            cv2.putText(new_frame, extra_text,
                        (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if pheromone_map is not None and pher_sat is not None:
            '''            pher_rgb = (cmap_pher((np.transpose(pheromone_map[f, :, :], (1, 0))) / (
                                    pheromone_map.max() + 0.0001)) * 255.0).astype(
                            'uint8')
                        pher_frame_canvas = cv2.resize(pher_rgb[:, :, 0:3], resolution)'''
            pher_frame_canvas = pher_rgb_resize[f, :, :, 0:3]
            pher_new_frame = overlay_transparent(pher_frame_canvas, new_frame_org_edge)
            pher_new_frame = overlay_transparent(pher_new_frame, new_frame_org)
            # Title
            cv2.putText(pher_new_frame, "Pheromone Map",
                        (100, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 50, 0), 2)
            new_frame = np.hstack((new_frame, pher_new_frame))
            # canvas_full[f, :, :, :] = np.hstack((canvas[f, :, :, :], canvas_temp_pher))

        frames.append(cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB))

    if video_output_path is None:
        return frames
    else:
        frames_to_video(frames=frames, output_path=video_output_path, fps=20)


def debug_display_2d(population, resolution, sim_width, sim_height, input_names, output_names, org_radius=3):
    from scipy.signal import fftconvolve

    graph_img = population.organisms[population.debug_target_org].brain.draw_graph(input_names=input_names,
                                                                                   output_names=output_names,
                                                                                   node_input_vals=population.debug_input_states,
                                                                                   node_output_vals=population.debug_output_states,
                                                                                   save_path=None, rank_order='LR')
    graph_img = cv2.resize(graph_img, (resolution[0], int(graph_img.shape[0] * resolution[0] / graph_img.shape[1])))

    no_frames = len(population.debug_pos)
    cmap_field = LinearSegmentedColormap.from_list('field cmap',
                                                   [[1, 0.3, 0], [0.5, 0.5, 0.5], [0, 1, 0.3]])  # field colormap
    org_fill_color = (128, 128, 128)
    org_edge_color = (0, 0, 0)

    p2 = np.zeros((org_radius * 4 + 1, org_radius * 4 + 1)).astype('uint8')
    p2 = cv2.circle(p2, (org_radius * 2, org_radius * 2), org_radius, 1, -1)
    p2 = p2 / p2.max()

    p3 = np.zeros((org_radius * 4 + 1, org_radius * 4 + 1)).astype('uint8')
    p3 = cv2.circle(p3, (org_radius * 2, org_radius * 2), org_radius, 1, 2)
    p3 = p3 / p3.max()

    if population.debug_field is not None:
        field_rgb = (cmap_field(np.transpose(population.debug_field, (1, 0))) * 255.0).astype(
            'uint8')
        field_rgb_resize = np.array(cv2.resize(field_rgb[:, :, 0:3], resolution))

    pos = np.array(population.debug_pos)
    coords = np.array([((pos[:, 0] + sim_width / 2) / sim_width * resolution[0]).astype('int'),
                       ((pos[:, 1] + sim_height / 2) / sim_height * resolution[1]).astype('int')])
    coords = np.transpose(coords, (1, 0))
    if population.debug_field is not None:
        new_frame_canvas = field_rgb_resize[:, :, 0:3]
    else:
        new_frame_canvas = (np.ones((resolution[0], resolution[1], 3)) * 255.0).astype('uint8')

    p1 = np.zeros(resolution)
    p1[coords[:, 1], coords[:, 0]] = 1

    cnv = fftconvolve(p1, p2, mode='same')
    cnv[cnv > 1] = 1
    cnv[cnv < 0] = 0
    # new_frame_org = np.flipud(np.fliplr(cnv))
    new_frame_org = cnv
    new_frame_org = np.tile(np.expand_dims(new_frame_org, 2), (1, 1, 4))
    new_frame_org[:, :, 0] *= org_fill_color[0]
    new_frame_org[:, :, 1] *= org_fill_color[1]
    new_frame_org[:, :, 2] *= org_fill_color[2]
    new_frame_org[:, :, 3] *= 255
    new_frame_org = new_frame_org.astype('uint8')

    # composite layers:
    new_frame = overlay_transparent(new_frame_canvas, new_frame_org)

    new_frame = cv2.circle(new_frame,
                           (coords[population.debug_target_org, 0], coords[population.debug_target_org, 1]),
                           org_radius + 2, (0, 0, 0), -1)
    new_frame = cv2.circle(new_frame,
                           (coords[population.debug_target_org, 0], coords[population.debug_target_org, 1]),
                           org_radius, (255, 255, 0), -1)

    # new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)

    frame_stacked = np.vstack((new_frame, graph_img))

    return frame_stacked


def metric_display_block(N_organisms, N_generations, metric, border_size=14, rescale_ratio=5,
                         block_color=(0, 127, 255), sample_intervals=1):
    # single pixel/block
    image = np.zeros((N_organisms, int(N_generations / sample_intervals), 3)).astype('uint8')
    image[:N_organisms, 0:int(metric.shape[1] / sample_intervals), 0] = (
            metric[:, range(0, metric.shape[1], sample_intervals)] * block_color[0]).astype('uint8')
    image[:N_organisms, 0:int(metric.shape[1] / sample_intervals), 1] = (
            metric[:, range(0, metric.shape[1], sample_intervals)] * block_color[1]).astype('uint8')
    image[:N_organisms, 0:int(metric.shape[1] / sample_intervals), 2] = (
            metric[:, range(0, metric.shape[1], sample_intervals)] * block_color[2]).astype('uint8')
    canvas = np.zeros(
        (N_organisms + 2 * border_size, int(N_generations / sample_intervals) + 2 * border_size, 3)).astype(
        'uint8') + 255
    canvas[border_size:(N_organisms + border_size),
    border_size:(int(N_generations / sample_intervals) + border_size)] = image
    canvas = cv2.resize(canvas, (int(canvas.shape[1] * rescale_ratio), int(canvas.shape[0] * rescale_ratio)),
                        interpolation=cv2.INTER_NEAREST)
    for i in range(0, N_generations + 1, 20 * sample_intervals):
        cv2.putText(canvas, '{}'.format(i),
                    ((border_size - 2 + int(i / sample_intervals)) * rescale_ratio, (border_size - 2) * rescale_ratio),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, '.'.format(i),
                    ((border_size + + int(i / sample_intervals)) * rescale_ratio, (border_size) * rescale_ratio),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 4, cv2.LINE_AA)
    return canvas


def get_average_position_roi(population_positions, data_charts, spread_ratio, trans_fee_pct):
    _, _, no_pos = population_positions.shape
    # Calculate ROI for average decision
    positions_avg = np.mean(population_positions, 1)
    positions_avg = positions_avg / np.tile(np.expand_dims(np.sum(positions_avg, 1), 1), (1, no_pos))
    fitness_avg = calc_fitness_roi_validation(data_charts=data_charts, positions=positions_avg,
                                              spread_ratio=spread_ratio, fee_pct=trans_fee_pct)
    return positions_avg, fitness_avg


def graph_preview(gen_no, organisms, static_input_names, output_names, save_path, data_input_count=None,
                  data_input_names=None,
                  position_density_names=None, graphs_grid_size=(3, 3)):
    if len(organisms) < 16:
        print("Skipped graph generation. Requires {} organisms, only {} provided".format(16, len(organisms)))
        return

    # Find the oldest organisms and plot their graphs
    ages = []
    for org in organisms:
        ages.append(org.age)
    ages_sort_idx = np.argsort(ages)
    organisms_select = []
    for i in range(1, 17):
        organisms_select.append(organisms[ages_sort_idx[-i]])

    max_dim = [0, 0]
    img_list = []
    # Draw graphs for each organism
    for i in range(graphs_grid_size[0]):
        for j in range(graphs_grid_size[1]):
            img = organisms_select[i * 4 + j].brain.draw_graph(static_input_names=static_input_names,
                                                               output_names=output_names,
                                                               save_path=None, data_input_count=data_input_count,
                                                               data_input_names=data_input_names,
                                                               position_density_names=position_density_names)
            if img.shape[0] > max_dim[0]:
                max_dim[0] = img.shape[0]
            if img.shape[1] > max_dim[1]:
                max_dim[1] = img.shape[1]
            img_list.append(img)

    # pad images
    for i in range(len(img_list)):
        v_pad = max_dim[0] - img_list[i].shape[0]
        h_pad = max_dim[1] - img_list[i].shape[1]
        img_list[i] = cv2.copyMakeBorder(img_list[i], int(v_pad / 2), int(v_pad / 2), int(h_pad / 2), int(h_pad / 2),
                                         cv2.BORDER_CONSTANT, value=(255, 255, 255))
        img_list[i] = cv2.resize(img_list[i], (max_dim[1], max_dim[0]))
        img_list[i] = cv2.copyMakeBorder(img_list[i], 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(50, 50, 50))

    canvas = None
    for i in range(graphs_grid_size[0]):
        row = None
        for j in range(graphs_grid_size[1]):
            if row is None:
                row = img_list[i * graphs_grid_size[0] + j]
            else:
                row = np.hstack((row, img_list[i * graphs_grid_size[0] + j]))
        if canvas is None:
            canvas = row
        else:
            canvas = np.vstack((canvas, row))
    cv2.imwrite(save_path + "/graph_gen_{}.png".format(gen_no), canvas)
    return canvas


def create_fan_chart(data, legend=False, ylim=None, bg_color=(0.2, 0.2, 0.2),
                     fan_offsets=(5, 10, 15, 20, 25, 30, 35, 40, 45),
                     population_configs=None, population_idx=None):
    """

    :param data:
    :param legend:
    :param ylim:
    :param bg_color:
    :param fan_offsets:
    :param population_configs:
    :param population_idx: If 'None', plot all populations
    :return:
    """
    if len(data.shape) == 2:
        data = np.expand_dims(data, 0)
    no_pop = data.shape[0]

    fig = plt.figure(figsize=(15, 10))
    fig.set_facecolor(bg_color)
    plt.style.use('dark_background')

    x = np.arange(data.shape[1])

    for p in range(no_pop):
        if population_idx is not None:
            p = population_idx
        if population_configs is None:
            c = np.array([0, 0, 1])
            label = '_nolegend_'
        else:
            c = population_configs[p].fan_color
            label = population_configs[p].pop_label
        data_pop = data[p, :, :]
        # for the median use `np.median` and change the legend below
        mean = np.mean(data_pop, axis=1)

        for offset in fan_offsets:
            low = np.percentile(data_pop, 50 - offset, axis=1)
            high = np.percentile(data_pop, 50 + offset, axis=1)
            # since `offset` will never be bigger than 50, do 55-offset so that
            # even for the whole range of the graph the fanchart is visible
            alpha = (55 - offset) / 100
            plt.fill_between(x, low, high, color=c, alpha=alpha, label='_nolegend_')
        if legend:
            plt.legend(['Mean'] + [f'Pct{2 * o}' for o in fan_offsets])
        if ylim is not None:
            plt.ylim(ylim)
        plt.plot(mean, color=c / 2, lw=2, label=label)
        if population_idx is not None:
            break

    plt.plot([0, x[-1]], [1, 1], linewidth=1, linestyle='--', c=(1, 1, 1), label='_nolegend_')
    # plt.grid()
    ax = plt.gca()
    ax.set_facecolor(bg_color)
    plt.style.use('default')

    return fig


def create_fanchart_plotly(dates, data, ylim=None, bg_color=(0.2, 0.2, 0.2),
                           fan_offsets=(5, 10, 15, 20, 25, 30, 35, 40, 45),
                           population_configs=None, population_idx=None, price_data=None):
    if len(data.shape) == 2:
        data = np.expand_dims(data, 0)
    no_pop = data.shape[0]

    fig = go.Figure()
    if price_data is not None:
        fig_specs = [[{"rowspan": 2}], [{}], [{}]]
        fig = make_subplots(rows=3, cols=1, specs=fig_specs, figure=fig,
                            subplot_titles=("Portfolio Performance", None, "BTC/USDT"), shared_xaxes=True)

    x = dates
    for p in range(no_pop):
        if population_idx is not None:
            p = population_idx
        if population_configs is None:
            c = np.array([0, 0, 1])
            label = '_nolegend_'
        else:
            c = population_configs[p].fan_color
            label = population_configs[p].pop_label
        data_pop = data[p, :, :]
        # for the median use `np.median` and change the legend below
        mean = np.mean(data_pop, axis=1)

        for offset in fan_offsets:
            low = np.percentile(data_pop, 50 - offset, axis=1)
            high = np.percentile(data_pop, 50 + offset, axis=1)
            # since `offset` will never be bigger than 50, do 55-offset so that
            # even for the whole range of the graph the fanchart is visible
            alpha = (55 - offset) / 100
            if price_data is not None:
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x, x[::-1]]),
                    y=np.concatenate([high, low[::-1]]),
                    fill='toself',
                    hoveron='points',
                    fillcolor='rgba({},{},{},{})'.format(c[0] * 255, c[1] * 255, c[2] * 255, alpha),
                    mode='none',
                    showlegend=False,
                ), row=1, col=1)
            else:
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x, x[::-1]]),
                    y=np.concatenate([high, low[::-1]]),
                    fill='toself',
                    hoveron='points',
                    fillcolor='rgba({},{},{},{})'.format(c[0] * 255, c[1] * 255, c[2] * 255, alpha),
                    mode='none',
                    showlegend=False,
                ))
        if price_data is not None:
            fig.add_trace(go.Scatter(x=x, y=mean, mode="lines",
                                     line=dict(color="rgb({},{},{})".format(c[0] * 128, c[1] * 128, c[2] * 128),
                                               dash="dash", width=3),
                                     name=label, showlegend=False), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=x, y=mean, mode="lines",
                                     line=dict(color="rgb({},{},{})".format(c[0] * 128, c[1] * 128, c[2] * 128),
                                               dash="dash", width=3),
                                     name=label, showlegend=False))
        if population_idx is not None:
            break

    if price_data is not None:
        fig.add_trace(go.Scatter(x=[x[0], x[-1]], y=[1, 1], line=dict(color="white", dash="dash"), showlegend=False),
                      row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=[x[0], x[-1]], y=[1, 1], line=dict(color="white", dash="dash"), showlegend=False))

    fig.update_layout(template="plotly_dark", title="ROI Fanchart",
                      font=dict(family="Courier New, monospace", size=18, color="PowderBlue"))
    fig.update_yaxes(range=ylim)

    if price_data is not None:
        fig.add_trace(go.Scatter(x=dates, y=price_data, mode='lines', name='BTC-USD'), row=3,
                      col=1)
        fig.layout['yaxis3']['range'] = [np.min(price_data) * 0.99, np.max(price_data) * 1.01]

    return fig


def plotly_positions_summary(dates, fitness_avg, fitness_bah, positions, price_data, currency_names, population_idx,
                             increments_seconds=180):
    from itertools import cycle
    from datetime import timedelta

    colors = px.colors.diverging.curl
    palette = cycle(colors)
    spec = [{"secondary_y": True}]
    _, T, no_pos = positions.shape
    fig = make_subplots(rows=no_pos, cols=1, subplot_titles=currency_names,
                        specs=[spec for n in range(no_pos)], shared_xaxes=True)

    for n in range(no_pos):
        fig.add_trace(go.Bar(x=dates[0:-1], y=positions[population_idx, :, n], name=currency_names[n],
                             width=timedelta(seconds=increments_seconds).seconds * 1000,
                             marker_line_width=0, marker_color=next(palette)), row=n + 1, col=1)
    for n in range(0, 2 * no_pos, 2):
        fig.update_layout({"yaxis{}".format(n + 1): {"range": [0, 1]}})

    for n in range(no_pos - 1):
        fig.add_trace(go.Scatter(x=dates[0:-1], y=price_data[n, :], name=currency_names[n]), row=n + 1, col=1,
                      secondary_y=True)

    fig.update_yaxes(range=[0, 1], secondary_y=False)

    roi = fitness_avg[population_idx, :]
    fig.add_trace(go.Scatter(x=dates[0:-1], y=roi, name="ROI"), row=no_pos, col=1,
                  secondary_y=True)
    fig.add_trace(go.Scatter(x=dates[0:-1], y=fitness_bah, name="ROI-BAH",
                             line=dict(color="rgb(204,204,204)", width=1, dash="dash")), row=no_pos, col=1,
                  secondary_y=True)
    fig.update_layout(
        {"yaxis{}".format(2 * no_pos): {"range": [roi.min() * 0.99, roi.max() * 1.01]}})

    fig.update_layout(template="plotly_dark", font_family="Droid Sans", height=2000)
    return fig


def plotly_genetic_summary(organisms):
    from src.genome import gene_codes, gene_type_labels

    gene_dict = dict()
    for ic, code in enumerate(gene_codes):
        gene_dict[code] = ic

    gene_type_count = np.zeros(len(gene_codes))

    #gene_pool_type = []
    for o in organisms:
        for g in o.genes:
            #gene_pool_type.append(int(g[0], 16))
            gene_type_count[gene_dict[g[0]]] += 1

    fig = go.Figure()

    fig.add_trace(go.Bar(x=gene_type_labels, y=gene_type_count,
                         marker_color='rgb(26, 118, 255)',
                         name='Gene Types'))

    fig.update_layout(template="plotly_dark", title="Gene Pool",
                      font=dict(family="Courier New, monospace", size=18, color="PowderBlue"))
    return fig


def plotly_summary_figures(dates, fitness_avg, fitness_org, fitness_bah, populations, price_data, pos_avg,
                           currency_names, population_idx=None, save_path=None, val_start=None):
    # Get yaxis limits for fanchart
    if population_idx is None:
        y_lim = [np.min([0.90, np.min(fitness_avg), fitness_org.min(2).mean(1).max(0)]),
                 np.max([1.1, np.max(fitness_avg), fitness_org.max(2).mean(1).max(0), fitness_bah.max()])]
    else:
        y_lim = [np.min([0.90, np.min(fitness_avg[population_idx, :]), fitness_org[population_idx].mean(1).max(0)]),
                 np.max([1.1, np.max(fitness_avg[population_idx, :]), fitness_org[population_idx].mean(1).max(0),
                         fitness_bah.max()])]

    comparison_price_data = 1 / price_data[9, :]
    fig_fanchart = create_fanchart_plotly(dates, np.transpose(fitness_org, (0, 2, 1)), ylim=y_lim,
                                          population_configs=populations, population_idx=population_idx,
                                          price_data=comparison_price_data)
    for p in range(len(populations)):
        if population_idx is not None:
            p = population_idx
        c = populations[p].line_color
        fig_fanchart.add_trace(go.Scatter(x=dates, y=fitness_avg[p, :],
                                          line=dict(color="rgb({},{},{})".format(c[0] * 256, c[1] * 256, c[2] * 256),
                                                    dash="dash", width=3),
                                          name="Average Position - {}".format(populations[p].pop_label)), row=1, col=1)
        if population_idx is not None:
            break
    fig_fanchart.add_trace(go.Scatter(x=dates, y=fitness_bah, line=dict(color="rgb(204,204,204)", width=2),
                                      name="BAH"), row=1, col=1)
    if val_start is not None:
        fig_fanchart.add_trace(go.Scatter(x=[dates[val_start], dates[val_start]], y=[0, 1.5],
                                 line=dict(color="rgb(255,0,0)", dash="dash", width=1),
                                 ))
    fig_positions = plotly_positions_summary(dates=dates, fitness_avg=fitness_avg, fitness_bah=fitness_bah,
                                             positions=pos_avg, price_data=price_data,
                                             currency_names=currency_names, population_idx=population_idx)

    fig_genome_summary = plotly_genetic_summary(organisms=populations[0].organisms)

    if save_path is not None:
        fig_fanchart.write_html(save_path + "/trading_fanchart.html")
        fig_positions.write_html(save_path + "/trading_positions.html")
        fig_genome_summary.write_html(save_path + "/genome_summary.html")
    else:
        fig_fanchart.show()


class Visualizer:
    def __init__(self, no_populations=1, resolution=None, sim_size=None, org_marker_size=3, field=None):
        super(Visualizer, self).__init__()

        self.no_populations = no_populations
        self.resolution = resolution
        self.sim_size = sim_size
        self.org_marker_size = org_marker_size

        # 2D plot arrays
        self.plot_pos = []
        self.plot_alive = []
        self.plot_health = []
        self.plot_pher = []
        self.plot_field = field

        # fitness tracking arrays
        self.fitness = []
        self.average_age = []
        self.genetic_diversity = []
        self.org_selection_count = []
        self.genes_match_count = []
        for i in range(self.no_populations):
            self.fitness.append([])
            self.average_age.append([])
            self.genetic_diversity.append([])
            self.org_selection_count.append([])
            self.genes_match_count.append([])

        # generate marker colors
        # marker_cmap = cm.plasma
        marker_cmap = cm.rainbow
        x = np.linspace(0.2, 1, self.no_populations)
        self.marker_c = []
        for i in range(self.no_populations):
            self.marker_c.append((np.array(marker_cmap(x[i])) * 255.0).astype('uint8'))

        # tracking plot arrays
        for i in range(self.no_populations):
            self.plot_pos.append([])
            self.plot_alive.append([])
            self.plot_health.append([])
            self.plot_pher.append([])

    def append_plot(self, population, env, pop_idx, USE_GPU=True):
        if USE_GPU:
            population.update_organism_arrays_gpu2cpu()
        pos = np.copy(np.array(population.pos))
        pos = (pos>0.1).astype('float')
        self.plot_pos[pop_idx].append(pos)
        self.plot_alive[pop_idx].append(np.copy(np.array(population.alive)))
        self.plot_health[pop_idx].append(np.copy(np.array(population.health)))
        if env.pher is not None:
            env.d_pher.copy_to_host(env.pher)
            self.plot_pher[pop_idx].append(np.copy(env.pher[population.pher_channel, :, :]))

    def reset_population_data(self):
        # reset stored positions
        self.plot_pos = []
        self.plot_alive = []
        self.plot_health = []
        self.plot_pher = []
        for i in range(self.no_populations):
            self.plot_pos.append([])
            self.plot_alive.append([])
            self.plot_health.append([])
            self.plot_pher.append([])

    def generate_2D_anim(self, gen_no, output_path, extra_text=None, fps=25, target_organism=None):

        org_edge_color = (0, 0, 0)
        org_dead_color = (0, 0, 0)
        cmap_field = LinearSegmentedColormap.from_list('field cmap',
                                                       [[0.01, 0.01, 0.01], [0.7, 0.3, 0.3],
                                                        [0, 1, 0.3]])  # field colormap
        cmap_pher = LinearSegmentedColormap.from_list('pheromone cmap', [[1, 1, 1], [0.5, 0, .5]])  # pheromone colormap

        circle_marker_kernel = np.zeros((self.org_marker_size * 4 + 1, self.org_marker_size * 4 + 1)).astype('uint8')
        circle_marker_kernel = cv2.circle(circle_marker_kernel, (self.org_marker_size * 2, self.org_marker_size * 2),
                                          self.org_marker_size, 1, -1)
        circle_marker_kernel = circle_marker_kernel / circle_marker_kernel.max()

        circle_marker2_kernel = np.zeros((self.org_marker_size * 4 + 1, self.org_marker_size * 4 + 1)).astype('uint8')
        circle_marker2_kernel = cv2.circle(circle_marker2_kernel, (self.org_marker_size * 2, self.org_marker_size * 2),
                                           self.org_marker_size, 1, 1)
        circle_marker2_kernel = circle_marker2_kernel / circle_marker2_kernel.max()

        layers = []

        if self.plot_field is not None:
            field_rgb = (cmap_field(np.transpose((self.plot_field + 1) / 2, (1, 0, 2))) * 255.0).astype(
                'uint8')
            field_rgb_resize = []
            for f in range(self.plot_field.shape[2]):
                frame = cv2.resize(field_rgb[:, :, f, :], self.resolution)
                if extra_text is not None:
                    cv2.putText(frame, extra_text,
                                (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, "Generation {}, T={}".format(gen_no, f),
                            (100, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                field_rgb_resize.append(frame)

            layers.append(field_rgb_resize)

        # organism markers
        for i in range(self.no_populations):
            pos = np.array(self.plot_pos[i])
            coords = np.array(
                [((pos[:, :, 0] + self.sim_size[0] / 2) / self.sim_size[0] * self.resolution[0]).astype('int'),
                 ((pos[:, :, 1] + self.sim_size[1] / 2) / self.sim_size[1] * self.resolution[1]).astype('int')])
            coords = np.transpose(coords, (1, 2, 0))
            frames = []
            for f in range(len(self.plot_pos[i])):
                points = np.zeros(self.resolution)
                points[coords[f, :, 1], coords[f, :, 0]] = 1

                frame = convolutional_render_points(coords=coords[f, :, :], marker_kernel=circle_marker_kernel,
                                                    marker_edge_kernel=circle_marker2_kernel,
                                                    resolution=self.resolution,
                                                    marker_c=self.marker_c[i], edge_c=org_edge_color,
                                                    alive_mask=self.plot_alive[i][f], dead_marker_c=org_dead_color)
                if target_organism is not None:
                    frame = cv2.circle(frame,
                                       (coords[f, target_organism, 0], coords[f, target_organism, 1]),
                                       4, (0, 0, 0, 255), -1)
                    frame = cv2.circle(frame,
                                       (coords[f, target_organism, 0], coords[f, target_organism, 1]),
                                       3, (255, 255, 255, 255), -1)
                    frame = cv2.circle(frame,
                                       (coords[f, target_organism, 0], coords[f, target_organism, 1]),
                                       2, (255, 255, 0, 255), -1)
                frames.append(frame)
            layers.append(frames)

        output_frames = overlay_layer_frames(layers, remove_alpha=True)

        if output_path is None:
            return output_frames
        else:
            frames_to_video(frames=output_frames, output_path=output_path, fps=fps, convert_bgr=True)

    # Numerical visualizations
    def numerical_position_bar_animation(self, data_charts, plot_population_idx=0, position_labels=None, no_bars=None,
                                         save_path=None,
                                         ffmpeg_path=None, fps=15, dpi=300):
        """
        Generate a simulation animation consisting of a subset of organism positions displayed as sections in a bar plot,
        with another window showing the position prices with an indicator of the current frame time.
        :param data_charts: Array [No Pos, T] consisting of the position data (eg. price data)
        :param plot_population_idx: Population number index (which population to generate preview for)
        :param position_labels: (Optional) List of string for labels of positions
        :param no_bars: Number of bars to plot, if None, plot all organisms (can be slow)
        :param save_path:  Path to save animation
        :param ffmpeg_path:  Path to ffmpeg.exe . If None, use environment variable 'ffmpeg'
        :param fps: FPS of output animation
        :param dpi: DPI (resolution) of output animation
        :return:
        """
        if ffmpeg_path is None:
            import os
            mpl.rcParams['animation.ffmpeg_path'] = os.environ['ffmpeg']
        else:
            mpl.rcParams['animation.ffmpeg_path'] = ffmpeg_path

        cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        positions = np.array(self.plot_pos[plot_population_idx])
        T, no_org, no_pos = positions.shape
        # normalize positions (to add =1 for each organism)
        positions = positions / np.tile(np.expand_dims(np.sum(positions, 2), 2), (1, 1, no_pos))

        if no_bars is None:
            bar_org_idx = range(no_org)
        else:
            bar_org_idx = list(np.linspace(0, no_org - 1, no_bars).astype('int'))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        data_charts = np.copy(data_charts)
        for i in range(no_pos - 1):
            data_charts[i, :] = (data_charts[i, :] - data_charts[i, :].min()) / (
                    data_charts[i, :].max() - data_charts[i, :].min()) * 2

        for ic in range(data_charts.shape[0]):
            u = ax2.plot(np.arange(T), data_charts[ic, 0:T], lw=2)

        animated_plots = []
        chart_t_line, = ax2.plot([], [], 'r-.', lw=2)
        animated_plots.append(chart_t_line)

        # intialize two plot objects (one in each axes)
        ax1.bar([], [])
        bar_width = 1
        bottom = np.zeros(len(bar_org_idx))
        for i in range(no_pos):
            bar = ax1.bar(range(len(bar_org_idx)), positions[0, bar_org_idx, i], bar_width, label=position_labels[i],
                          bottom=bottom, color=cycle_colors[i])
            bottom += positions[0, bar_org_idx, i]
            for b in bar:
                animated_plots.append(b)

        ax1.set_xlim(0, len(bar_org_idx) - 1)
        ax1.set_ylim(0, 1)
        ax1.set_xticks(range(len(bar_org_idx)), list(np.array(bar_org_idx) + 1))
        # ax1.legend()

        ax2.set_ylim(0, 2)
        ax2.set_xlim(0, T)
        ax2.grid()

        def run(frame_no):
            # update the data
            x_t_line = frame_no
            h = positions[frame_no, :, :]

            animated_plots[0].set_data(x_t_line, ax2.get_ylim())
            bottom_offset = np.zeros(len(bar_org_idx))
            for i in range(no_pos):
                for j in range(len(bar_org_idx)):
                    animated_plots[j + i * len(bar_org_idx) + 1].set_height(h[j, i])
                    animated_plots[j + i * len(bar_org_idx) + 1]._y0 = bottom_offset[j]
                    bottom_offset[j] += h[j, i]
            return animated_plots

        anim = animation.FuncAnimation(fig=fig, func=run, frames=T, blit=False, interval=10,
                                       repeat=False)
        if save_path is not None:
            writervideo = animation.FFMpegWriter(fps=fps)
            anim.save(save_path, writer=writervideo, dpi=dpi)
        else:
            plt.show()

    def numerical_position_roi_preview(self, data_charts, fitness_bah, plot_organism_idx=(0, None),
                                       plot_population_idx=0, save_path=None, position_labels=None, figsize=(10, 10),
                                       y_scale_mode="daily", spread_data=None, fee_pct=None):
        """
        Generate a detailed plot showing a line plot with the price and a bar plot with the position ratio for each
        currency. Do this for a specified organism in the first coloum and the average position in the second.
        :param data_charts:
        :param fitness_bah:
        :param plot_organism_idx:
        :param plot_population_idx:
        :param save_path:
        :param position_labels:
        :param figsize:
        :param y_scale_mode:
        :param spread_data:
        :param fee_pct:
        :param fee_discount_idx:
        :return:
        """
        positions = normalize_positions(self.plot_pos[plot_population_idx])
        T, no_org, no_pos = positions.shape

        pos_c = (.9, 0.5, 0)  # color for position bars
        usd_c = (0, 0.8, .5)  # color for USD

        fig = plt.figure(figsize=(figsize[0], figsize[1]))

        for i_org in range(len(plot_organism_idx)):

            if plot_organism_idx[i_org] is None:
                pos, fitness = get_average_position_roi(population_positions=positions, data_charts=data_charts,
                                                        spread_ratio=spread_data,
                                                        trans_fee_pct=fee_pct)
            else:
                pos = positions[:, plot_organism_idx[i_org], :]
                fitness = calc_fitness_roi_validation(data_charts=data_charts, positions=pos, spread_ratio=spread_data,
                                                      fee_pct=fee_pct)

            roi_y_lim = None
            if y_scale_mode == "daily":
                roi_y_lim = [0, fitness.max() * 1.1]
            elif y_scale_mode == "intraday":
                roi_y_lim = [fitness.min() * 0.9, fitness.max() * 1.02]

            ax_usd = fig.add_subplot(no_pos, len(plot_organism_idx), 1 + i_org, label="usd")
            ax_fit = fig.add_subplot(no_pos, len(plot_organism_idx), 1 + i_org, label="fitness", frame_on=False)
            ax_fit.plot([0, T], [1, 1], 'k--', alpha=0.5, label='_nolegend_')
            ax_fit.plot(range(T), fitness, linewidth=2, c='k', label='Fitness (ROI)')
            ax_fit.plot(range(T), fitness_bah, linewidth=1, c='k', linestyle='--', label='Fitness BAH (ROI)')
            ax_fit.set_xlim([0, T])
            ax_usd.set_xlim([0, T])
            ax_fit.set_ylim(roi_y_lim)
            ax_usd.set_ylim([0, 2.5])
            ax_usd.tick_params(axis='y', colors=usd_c)
            ax_usd.yaxis.tick_right()
            # ax_fit.grid()
            ax_usd.bar(range(T), pos[:, -1], color=usd_c, edgecolor=usd_c)

            # Show only y ticks up to y=1.0
            y_ticks = ax_usd.get_yticks()
            y_ticks = y_ticks[y_ticks <= 1.0]
            ax_usd.set_yticks(y_ticks)

            if i_org == 0:
                ax_fit.set_ylabel('ROI', color='k')
            if len(self.fitness[plot_population_idx]) > 0:
                if plot_organism_idx[i_org] is None:
                    ax_fit.title.set_text(
                        'Population Mean (fitness={})'.format(plot_organism_idx[i_org], fitness[-1]))
                else:
                    ax_fit.title.set_text(
                        'Organism {} (fitness={})'.format(plot_organism_idx[i_org],
                                                          self.fitness[plot_population_idx][-1][
                                                              plot_organism_idx[i_org]]))
            else:
                if plot_organism_idx[i_org] is None:
                    ax_fit.title.set_text(
                        'Population Mean'.format(plot_organism_idx[i_org]))
                else:
                    ax_fit.title.set_text('Organism {}'.format(plot_organism_idx[i_org]))

            for i_pos in range(no_pos - 1):

                # ax1 is the current position bar chart (0->1)
                ax1 = fig.add_subplot(no_pos, len(plot_organism_idx), 1 + (1 + i_pos) * len(plot_organism_idx) + i_org,
                                      label="1")
                # ax2 is the position/currnecy value line chart
                ax2 = fig.add_subplot(no_pos, len(plot_organism_idx), 1 + (1 + i_pos) * len(plot_organism_idx) + i_org,
                                      label="2", frame_on=False)
                value_chart_ylim = None
                if y_scale_mode == "daily":
                    value_chart_ylim = [-0.5, np.max(data_charts[i_pos, 0:T]) * 1.1]
                elif y_scale_mode == "intraday":
                    value_chart_ylim = [np.min(data_charts[i_pos, 0:T]) * 0.85, np.max(data_charts[i_pos, 0:T]) * 1.05]

                ax1.bar(range(T), pos[:, i_pos], color=pos_c, edgecolor=pos_c)
                ax1.yaxis.tick_right()
                ax1.yaxis.tick_right()
                # ax1.set_ylabel('position', color=pos_c)
                # ax1.yaxis.set_label_position('right')
                ax1.tick_params(axis='y', colors=pos_c)
                ax1.set_ylim([0, 2.5])
                # Show only y ticks up to y=1.0
                y_ticks = ax1.get_yticks()
                y_ticks = y_ticks[y_ticks <= 1.0]
                ax1.set_yticks(y_ticks)

                ax2.plot(np.arange(T), data_charts[i_pos, 0:T], linewidth=2)
                # ax2.plot([0, T], [1, 1], 'k--', alpha=0.5)

                ax2.set_ylim(value_chart_ylim)
                ax2.grid()
                if i_org == 0:
                    if position_labels is not None:
                        ax2.set_ylabel(position_labels[i_pos], color="b")
                    else:
                        ax2.set_ylabel("price", color="b")
                ax1.set_xlim([0, T])
                ax2.set_xlim([0, T])
                ax1.set_xticks([])

        if save_path is not None:
            plt.savefig(save_path)

        plt.close()
        return fig

    def numerical_BAH_comparison(self, fitness_bah, fitness_avg, fitness_org, position_avg, population_idx=0,
                                 save_path=None):
        """
        For a given currency dataset and population positions, compute the BAH fitness and fitness for each organism.
        For each organism, calculate the ROI/BAH_ROI ratio, and export as histogram data for tensorboard. Also calculate
        the average, maximum and minimum ROI/ROI_BAH. The average is the ROI of the average position, not the average
        ROI.
        :param data_charts: currency data [no correncies, samples]
        :param population_idx: index of population to use of all available populations
        :param save_path: path to save histogram. If set to "None", display the plot. If set to "tensorboard", export
            the histogram data for tensorboard to plot the histogram.
        :return: Either the histogram figure, or the histogram data
        """
        plt.close()

        no_pop, no_org, T = fitness_org.shape

        bah_comparison = []
        for i_org in range(no_org):
            bah_comparison.append(fitness_org[population_idx, i_org, -1] / fitness_bah[-1])

        # Calculate bah comparison metrics
        metrics = {"ROI/BAH_ROI Mean": np.mean(np.array(bah_comparison)),
                   "ROI/BAH_ROI Max": np.max(np.array(bah_comparison)),
                   "Average Decision ROI": fitness_avg[population_idx][-1],
                   "Switching Activity": np.mean(np.var(position_avg[population_idx, :, :], 0))
                   }

        if save_path == "tensorboard":
            return bah_comparison, metrics
        else:
            f = plt.figure(figsize=(10, 10))
            plt.hist(bah_comparison, 35)
            plt.grid()
            plt.xlabel('ROI/BAH_ROI')
            plt.ylabel('Count')
            if save_path is None:
                plt.show()
            else:
                plt.savefig(save_path)
            plt.close()
            return f, metrics

    def numerical_roi_fanchart(self, fitness_avg, fitness_org, fitness_bah, populations, save_path=None,
                               y_scale_mode="daily", renderer="mpl", dates=None, population_idx=None, val_start=None):
        """
        Generate a fan chart (distribution line plot) of the population ROI. Include the BAH ROI and the ROI of the
        average population decision as separate lines.
        :param population_idx:
        :param data_charts:
        :param plot_population_idx: If 'None', plot all populations
        :param save_path: Path to save figure. If "None", skip saving
        :param y_scale_mode: Scale of y-axis in plot. Either "daily" for large scale [0.4->3.5], or "intraday" for small
            scale [0.94->1.07]
        :return: pyplot figure object
        """

        y_lim = None
        if y_scale_mode == "daily":
            y_lim = [0.4, 3.5]
        elif y_scale_mode == "intraday":
            if population_idx is None:
                y_lim = [np.min([0.90, np.min(fitness_avg), fitness_org.min(2).mean(1).max(0)]),
                         np.max([1.1, np.max(fitness_avg), fitness_org.max(2).mean(1).max(0), fitness_bah.max()])]
            else:
                y_lim = [
                    np.min([0.90, np.min(fitness_avg[population_idx, :]), fitness_org[population_idx].mean(1).max(0)]),
                    np.max([1.1, np.max(fitness_avg[population_idx, :]), fitness_org[population_idx].mean(1).max(0),
                            fitness_bah.max()])]

        label_str = "Average ROI"
        if len(populations) > 1:
            label_str = "{} - Average ROI"

        # Should maybe abstract this to just having color/label inputs rather than population classes
        if renderer == "mpl":
            plt.close()
            fig = create_fan_chart(np.transpose(fitness_org, (0, 2, 1)), ylim=y_lim, population_configs=populations,
                                   population_idx=population_idx)
            if population_idx is None:
                for p in range(len(populations)):  # Plot lines for average decisions in each population
                    plt.plot(fitness_avg[p, :], linestyle="--", color=populations[p].line_color,
                             label=label_str.format(populations[p].pop_label))
            else:
                plt.plot(fitness_avg[population_idx, :], linestyle="--", color=populations[population_idx].line_color,
                         label=label_str.format(populations[population_idx].pop_label))
            plt.plot(fitness_bah, linestyle='dashed', color=(0.8, 0.8, 0.8), label="BAH ROI")
            plt.legend(loc='upper left')
            plt.title("ROI Fanchart vs Buy-and-Hold")
            if save_path is not None:
                plt.savefig(save_path)
        elif renderer == "plotly":
            fig = create_fanchart_plotly(dates, np.transpose(fitness_org, (0, 2, 1)), ylim=y_lim,
                                         population_configs=populations)
            for p in range(len(populations)):  # Plot lines for average decisions in each population
                c = populations[p].line_color
                fig.add_trace(go.Scatter(x=dates, y=fitness_avg[p, :],
                                         line=dict(color="rgb({},{},{})".format(c[0] * 256, c[1] * 256, c[2] * 256),
                                                   dash="dash", width=3),
                                         name=label_str.format(populations[p].pop_label)))
            fig.add_trace(go.Scatter(x=dates, y=fitness_bah, line=dict(color="rgb(204,204,204)", width=4),
                                     name="BAH"))
            if val_start is not None:
                fig.add_trace(go.Scatter(x=[dates[val_start], dates[val_start]], y=[0, 1.5],
                                         line=dict(color="rgb(255,0,0)", dash="dash", width=1),
                                         ))
            if save_path is not None:
                fig.write_html(save_path)
        else:
            print("ERROR (Visualizer.numerical_roi_fanchart()): Invalid renderer option \"{}\"".format(renderer))
            return None

        return fig

    def log_numerical(self, data_charts, organism_idx, save_path=None, population_idx=0):

        positions = np.array(self.plot_pos[population_idx])
        T, no_org, no_pos = positions.shape
        positions_norm = positions / np.tile(np.expand_dims(np.sum(positions, 2), 2), (1, 1, no_pos))

        if len(self.fitness[population_idx]) > 0:
            log_str = "Organism {}: Recorded fitness = {:0.3f}\n".format(organism_idx,
                                                                         self.fitness[population_idx][-1][organism_idx])
        else:
            log_str = "Organism {}\n".format(organism_idx)

        fitness_current = 1.
        for t in range(T):
            log_str += "t={}: fitness_old={:0.3f}, ".format(t, fitness_current)
            log_str += "positions: {}, ".format(positions[t, organism_idx, :])
            log_str += "position_value = ["
            for i_pos in range(no_pos):
                log_str += "{:0.2f}, ".format(data_charts[i_pos, t])
            log_str += "], "
            log_str += "roi: ["
            for i_pos in range(no_pos):
                log_str += "{:0.1f}*{:0.2f}, ".format(positions_norm[t, organism_idx, i_pos],
                                                      data_charts[i_pos, t + 1] / data_charts[i_pos, t])

            roi = 0
            for i_pos in range(no_pos):
                roi += data_charts[i_pos, t + 1] / data_charts[i_pos, t] * positions_norm[t, organism_idx, i_pos]
            fitness_current *= roi

            log_str += "], ROI={:0.3f}, fitness_new={:0.3f}\n".format(roi, fitness_current)

        if save_path is not None:
            with open(save_path, 'w') as f:
                f.write(log_str)
        else:
            print(log_str)

    # Genetic visualizations
    def generate_summary_plots(self, gene_pool, plot_dir, figure_size=(20, 20), template=None, no_internal_nodes=None,
                               population_idx=0):
        if gene_pool is not None and False:
            genetic_histogram(gene_pool=gene_pool, save_path=plot_dir + "gene_hist.png",
                              save_text_file=plot_dir + "top_genes.txt", template=template,
                              no_internal_nodes=no_internal_nodes)

        plt.close()
        plot_created = False

        if len(self.fitness[population_idx]) > 0:
            # Fitness line plots
            f, ax = plt.subplots(2, 1, figsize=figure_size)
            fitness_sorted = np.sort(self.fitness[population_idx])

            smooth_steps = 20
            if fitness_sorted.shape[0] < smooth_steps:
                smooth_steps = fitness_sorted.shape[0]
            elif fitness_sorted.shape[0] > 100:
                smooth_steps = 50

            fitness_avg_smooth = ema(np.mean(fitness_sorted, 1), smooth_steps)
            fitness_T15_smooth = ema(np.mean(fitness_sorted[:, -int(fitness_sorted.shape[1] * 0.15):], 1), smooth_steps)
            fitness_L15_smooth = ema(np.mean(fitness_sorted[:, :int(fitness_sorted.shape[1] * 0.15)], 1), smooth_steps)
            fitness_max_smooth = ema(fitness_sorted[:, -1], smooth_steps)

            ax[0].plot(np.mean(fitness_sorted, 1), 'k', label="_nolegend_", alpha=0.10)  # Avg
            ax[0].plot(np.mean(fitness_sorted[:, 0:int(fitness_sorted.shape[1] * 0.10)], 1), 'r',
                       label="_nolegend_", alpha=0.2)  # Top 15%
            ax[0].plot(np.mean(fitness_sorted[:, int(fitness_sorted.shape[1] * 0.10):], 1), 'b',
                       label="_nolegend_", alpha=0.2)  # Lowest 15%
            ax[0].plot(fitness_sorted[:, -1], 'g', label="_nolegend_", alpha=0.10)  # Maximum
            ax[0].plot([0, len(fitness_sorted)], [1., 1.], c='k', linestyle='--', label="_nolegend_")
            # Plot smoothed lines
            ax[0].plot(fitness_avg_smooth, 'k', label="Fitness Average")
            ax[0].plot(fitness_T15_smooth, 'r', label="Fitness Top 15%")
            ax[0].plot(fitness_L15_smooth, 'b', label="Fitness Lowest 15%")
            ax[0].plot(fitness_max_smooth, 'g', label="Fitness Maximum")

            ax[0].legend(loc='lower right')
            ylim_lower = np.min(
                np.array([fitness_avg_smooth, fitness_T15_smooth, fitness_L15_smooth, fitness_max_smooth]) * 0.9)
            ax[0].set_ylim([ylim_lower, fitness_sorted.max()])
            ax[0].grid()

            plot_created = True
            ax[0].title.set_text("Population Fitness")
            # Genetic diversity plots
            if len(self.genetic_diversity[population_idx]) > 0:
                ax[1].plot(self.genetic_diversity[population_idx])
                ax[1].set_ylim([0, np.max(self.genetic_diversity[population_idx]).max() * 1.05])
                ax[1].grid()
                ax[1].title.set_text("Genetic Diversity")
                plot_created = True
        # NOT ACTIVE ******
        if plot_created and False:
            pickle.dump([np.mean(self.fitness[population_idx], 1), self.genetic_diversity[population_idx]],
                        open(plot_dir + "data.p", 'wb'))

        if plot_created:
            plt.savefig(plot_dir + "average_fitness.png")
        plt.close()

    def generate_fitness_block_plot(self, plot_dir, N_generations, population_idx=0):
        """
        Generate a block-pixel diagram of sorted organism fitnesses. Birghter color represents higher fitness,

        (Should only display every N generations to avoid massive file sizes)
        :param plot_dir:
        :param N_generations:
        :param population_idx:
        :param block_intervals:
        :return:
        """
        # Determine how many generations to skip between samples (to avoid massively long image files)
        block_intervals = 1
        if N_generations > 10000:
            block_intervals = 30
        if N_generations > 1000:
            block_intervals = 10
        elif N_generations > 500:
            block_intervals = 2

        if (len(self.fitness[population_idx]) / block_intervals) > 1:
            N_organisms = len(self.fitness[population_idx][0])
            fitness = np.array(self.fitness[population_idx])
            sort_idx = np.argsort(fitness, axis=1)
            fitness_sorted = np.take_along_axis(fitness, sort_idx, axis=1)
            fitness_sorted = np.transpose(fitness_sorted, (1, 0))[::-1]
            fitness_sorted = (fitness_sorted - fitness_sorted.min()) / (fitness_sorted.max() - fitness_sorted.min())
            image_fit = metric_display_block(N_organisms, N_generations, fitness_sorted,
                                             sample_intervals=block_intervals)

            if len(self.org_selection_count[population_idx]) > 0:
                sel_count_sorted = np.take_along_axis(np.array(self.org_selection_count[0]), sort_idx, axis=1)
                sel_count_sorted = np.transpose(sel_count_sorted, (1, 0))[::-1]
                sel_count_sorted = (sel_count_sorted - sel_count_sorted.min()) / (
                        sel_count_sorted.max() - sel_count_sorted.min())
                image_sel = metric_display_block(N_organisms, N_generations, sel_count_sorted,
                                                 sample_intervals=block_intervals)

                image = np.vstack((image_fit, image_sel))
            else:
                image = image_fit

            cv2.imwrite(plot_dir + "fitness_pixel_diagram.png", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def key_connection_genes_update(self, organisms, connection_list):
        for i in range(self.no_populations):
            # for each organism, find how many gene match the criteria. Provide a number 0->1 for number of matches
            new_genes_match_count = []
            for j in range(len(organisms)):
                has_genes, weights = check_genome_has_connections(organism=organisms[j],
                                                                  in_out_idx_list=connection_list)
                new_genes_match_count.append(np.sum(has_genes) / len(has_genes))
            self.genes_match_count[i].append(new_genes_match_count)

    def key_connection_genes_block_plot(self, plot_dir, N_generations):
        for i in range(self.no_populations):
            if len(self.fitness[i]) > 0:
                N_organisms = len(self.fitness[i][0])
                fitness = np.array(self.fitness[i])
                sort_idx = np.argsort(fitness, axis=1)
                # Sort the matched genes metric by fitness so the highest fitness organism appear at the top
                genes_match_count_sorted = np.transpose(
                    np.take_along_axis(np.array(self.genes_match_count[i]), sort_idx, axis=1), (1, 0))[::-1]
                image = metric_display_block(N_organisms, N_generations, genes_match_count_sorted,
                                             block_color=(255, 128, 0))
                cv2.imwrite(plot_dir + "debug_gene_match_pixel_diagram.png", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def numerical_visualization_set(visualizer, populations, env, path_config, g, no_generations, sim_type, writer,
                                gene_pool, config, data_interval_mode="daily", mode="train",
                                key_debug_gene_connections=None, game_track=None):
    # Summary output of metrics
    metrics = dict()
    if mode == "train" or mode == "test":
        metrics["Generations Trained"] = g
    if config['mode'] == 'none':
        return metrics

    writer_pop_prefix = "{}-{} "  # prefix to add to tensorboard summarywriter labels
    writer_category = "Train"
    img_mode_prefix = ""
    if mode == "validate":
        writer_category = "Validation"
        img_mode_prefix = "val"
        g = "val"
    elif mode == "test":
        writer_category = "Test"
        img_mode_prefix = "Test"

    if mode == "validate" and np.array(visualizer.plot_pos[0]).shape[0] > 2000:
        pbar = tqdm(total=(len(populations)) * 4 + 1)
    else:
        pbar = None

    if config['animation'] and mode == "train":
        if sim_type == "2D":
            visualizer.generate_2D_anim(gen_no=g,
                                        output_path=path_config.anim_preview_dir + "/sim_anim_{}.mp4".format(g),
                                        fps=15, extra_text="Fitness: avg {:0.2f}, max {:0.2f}".format(
                    np.mean(populations[0].fitness),
                    populations[0].fitness.max()))
        elif sim_type == "numerical":
            # Generate bar plot animation
            print("numerical plot")
            visualizer.numerical_position_bar_animation(data_charts=env.data_out,
                                                        position_labels=["BTC", "ETH", "LTC", "USD"],
                                                        no_bars=20,
                                                        save_path=path_config.anim_preview_dir + "/sim_anim_{}.mp4".format(
                                                            g))

    # If numerical simulation, generate ROI plot preview and activity log
    if sim_type == "numerical":
        # Text file containing population positions and input data for each time step
        roi_log_suffix = ""
        if mode == "validate":
            roi_log_suffix = "_val"
        elif mode == "test":
            roi_log_suffix = "_test"

        # Precalculate the ROI performance of populations and buy-and-hold
        positions = normalize_positions(visualizer.plot_pos)
        no_pop, T, no_org, no_pos = positions.shape

        bah_positions = np.zeros(no_pos)
        bah_positions[0:(no_pos - 1)] = 1 / (no_pos - 1)
        bah_positions = np.round(bah_positions, 2)
        bah_positions[-1] = 1. - np.sum(bah_positions)
        bah_positions = np.tile(bah_positions, (T, 1))
        roi_bah = calc_fitness_roi_validation(data_charts=env.data_out, positions=bah_positions)
        positions_avg = []
        roi_avg_positions = []
        roi_organisms = []

        pos_avg = []
        for p in range(len(populations)):
            pos, fit_avg = get_average_position_roi(population_positions=positions[p], data_charts=env.data_out,
                                                    spread_ratio=env.spread_ratio if populations[
                                                        p].use_spread else None,
                                                    trans_fee_pct=env.trading_fees if populations[p].use_trans_fee else None)
            pos_avg.append(pos)
            positions_avg.append(pos)
            roi_avg_positions.append(fit_avg)
            fitness_track_pop = []
            # if T < 1500:
            iterator = range(no_org)
            # else:
            #    print("Calculating population {} organism ROIs".format(p))
            #    iterator = tqdm(range(0, no_org, 1))
            for i_org in iterator:
                fitness = calc_fitness_roi_validation(data_charts=env.data_out, positions=positions[p, :, i_org, :],
                                                      fee_pct=env.trading_fees if populations[p].use_trans_fee else None,
                                                      spread_ratio=env.spread_ratio if populations[
                                                          p].use_spread else None,
                                                      debug=False)
                fitness_track_pop.append(fitness)
            roi_organisms.append(fitness_track_pop)
        pos_avg = np.array(pos_avg)
        roi_avg_positions = np.array(roi_avg_positions)
        roi_organisms = np.array(roi_organisms)
        metrics["population final ROI"] = list(roi_organisms[:, -1, :].mean(1))

        # Generate Plots using calculated ROIs
        for p in range(len(populations)):
            if config['performance log']:
                visualizer.log_numerical(data_charts=env.data_out, organism_idx=0, population_idx=0,
                                         save_path=path_config.log_dir + "/roi_log" if len(populations) == 1 else
                                         path_config.pop_paths[p].log_dir + "/roi_log" + roi_log_suffix + ".txt")

            if pbar is not None:
                pbar.update(1)

            if mode == "train" or mode == "validate":
                # Summary of positions across time in subplots
                # Positions are displayed as bar charts, with ROI as a line
                if config["numerical positions"]:
                    fig_roi = visualizer.numerical_position_roi_preview(data_charts=env.data_out,
                                                                        plot_organism_idx=[0, None],
                                                                        plot_population_idx=p,
                                                                        fitness_bah=roi_bah,
                                                                        position_labels=env.numerical_position_labels,
                                                                        save_path=path_config.anim_preview_dir + "/roi_{}.png".format(
                                                                            g) if len(populations) == 1 else
                                                                        path_config.pop_paths[
                                                                            p].anim_dir + "/roi_{}.png".format(g),
                                                                        figsize=(50, 25),
                                                                        y_scale_mode=data_interval_mode,
                                                                        spread_data=env.spread_ratio if populations[
                                                                            p].use_spread else None)

                    writer.add_figure(tag=writer_category + "/" + writer_pop_prefix.format(p, populations[
                        p].pop_label) + "Sample_ROI_Charts", figure=fig_roi, global_step=0 if mode == "validate" else g)

                if pbar is not None:
                    pbar.update(1)
            if mode == "validate" or mode == "test":
                # Calculate the BAH population performance histogram
                bah_comparison, bah_compare_metric = visualizer.numerical_BAH_comparison(fitness_bah=roi_bah,
                                                                                         fitness_avg=roi_avg_positions,
                                                                                         fitness_org=roi_organisms,
                                                                                         position_avg=pos_avg,
                                                                                         save_path="tensorboard")
                writer.add_scalar(
                    writer_category + "/" + writer_pop_prefix.format(p, populations[p].pop_label) + "ROI|ROI_BAH Mean",
                    bah_compare_metric['ROI/BAH_ROI Mean'], 0 if mode == "validate" else g)
                writer.add_histogram(
                    tag=writer_category + "/" + writer_pop_prefix.format(p, populations[p].pop_label) + "ROI|ROI_BAH",
                    values=bah_comparison, global_step=0 if mode == "validate" else g)
                metrics["BAH Compare"] = bah_compare_metric
                if pbar is not None:
                    pbar.update(1)

        if config["plotly summary"]:
            plotly_summary_figures(dates=env.data_dates, fitness_avg=roi_avg_positions, fitness_org=roi_organisms,
                                   fitness_bah=roi_bah, populations=populations, price_data=env.data_out,
                                   pos_avg=pos_avg, currency_names=env.env_data_file['currencies'],
                                   population_idx=0, val_start=env.validation_start_sample,
                                   save_path=path_config.gs_state_dir + "/")

            fig_roi = visualizer.numerical_roi_fanchart(dates=env.data_dates, fitness_avg=roi_avg_positions,
                                                        fitness_org=roi_organisms,
                                                        fitness_bah=roi_bah, populations=populations,
                                                        y_scale_mode=data_interval_mode,
                                                        save_path="./" + img_mode_prefix + "_ROI_fanchart_plotly.png",
                                                        renderer="plotly", val_start=env.validation_start_sample)


            fig_roi.write_html(path_config.gs_state_dir + "/" + img_mode_prefix + "_ROI_fanchart.html")
            # ADD REMAINING PLOTLY SUMMARY PLOTS HERE
        elif mode == "validate" or mode == "test":

            # Generate the ROI fanchart
            fanchart_save_dir = path_config.plot_dir
            if mode == "test":
                fanchart_save_dir = path_config.gs_state_dir
            fig_roi = visualizer.numerical_roi_fanchart(fitness_avg=roi_avg_positions, fitness_org=roi_organisms,
                                                        fitness_bah=roi_bah, populations=populations,
                                                        y_scale_mode=data_interval_mode,
                                                        save_path=fanchart_save_dir + "/" + img_mode_prefix + "_ROI_fanchart.png",
                                                        population_idx=0)

            # When using test, also copy the fanchart to the samples folder
            if mode == "test":
                shutil.copy(src=fanchart_save_dir + "/" + img_mode_prefix + "_ROI_fanchart.png",
                            dst=path_config.samples_dir + "/" + img_mode_prefix + "_ROI_fanchart.png")
            if mode == "validate":
                shutil.copy(src=fanchart_save_dir + "/" + img_mode_prefix + "_ROI_fanchart.png",
                            dst=path_config.samples_dir + "/" + img_mode_prefix + "_ROI_fanchart_validation.png")
                shutil.copy(src=fanchart_save_dir + "/" + img_mode_prefix + "_ROI_fanchart.png",
                            dst=path_config.gs_state_dir + "/Validation_ROI_fanchart.png")

            writer.add_figure(tag=writer_category + "/ROI Fan Chart", figure=fig_roi,
                              global_step=g if mode == "test" else 0)

            if pbar is not None:
                pbar.update(1)
        for p in range(len(populations)):
            train_graphs = graph_preview(gen_no=g, organisms=populations[p].organisms,
                                         static_input_names=populations[p].template.input_names,
                                         output_names=populations[p].template.output_names,
                                         save_path=path_config.graph_preview_dir if len(populations) == 1 else
                                        path_config.pop_paths[p].graph_dir,
                                         data_input_count=populations[p].template.data_input_count,
                                         data_input_names=env.data_in_names,
                                         position_density_names=env.position_density_names)
            writer.add_image(
                tag=writer_category + "/" + writer_pop_prefix.format(p, populations[p].pop_label) + "Decision_Graphs",
                img_tensor=np.transpose(train_graphs, (2, 0, 1)),
                global_step=g if (mode == "test" or mode == "train") else 0)
            if pbar is not None:
                pbar.update(1)
            if mode == "train":
                visualizer.generate_summary_plots(gene_pool=gene_pool,
                                                  plot_dir=path_config.plot_dir if len(populations) == 1 else
                                                  path_config.pop_paths[p].plot_dir,
                                                  template=populations[p].template,
                                                  no_internal_nodes=populations[p].node_counts["hidden"],
                                                  population_idx=p)
                visualizer.generate_fitness_block_plot(
                    plot_dir=path_config.plot_dir if len(populations) == 1 else path_config.pop_paths[p].plot_dir,
                    N_generations=no_generations, population_idx=p)
    elif sim_type == "game":
        for p in range(len(populations)):
            train_graphs = graph_preview(gen_no=g, organisms=populations[p].organisms,
                                         static_input_names=populations[p].template.input_names,
                                         output_names=populations[p].template.output_names,
                                         save_path=path_config.graph_preview_dir if len(populations) == 1 else
                                        path_config.pop_paths[p].graph_dir,
                                         data_input_count=populations[p].template.data_input_count,
                                         data_input_names=env.data_in_names,
                                         position_density_names=env.position_density_names)
        if game_track is not None:
            game_track.generate_animation(path_config.anim_preview_dir + "game_{}.mp4".format(g))




    ## NEED TO ADAPT FOR MULTIPLE POPULATIONS
    if key_debug_gene_connections is not None:
        visualizer.key_connection_genes_block_plot(plot_dir=path_config.plot_dir,
                                                   N_generations=no_generations)

    return metrics


def numerical_visualization_set_batch(visualizer, populations, env, path_config, g, sim_type, writer,
                                      mode="train", skip_animation=True):
    """
    For use when using 'batch' mode (matplotlib functions inaccessible)
    :param visualizer:
    :param populations:
    :param env:
    :param path_config:
    :param g:
    :param no_generations:
    :param sim_type:
    :param writer:
    :param gene_pool:
    :param data_interval_mode:
    :param mode:
    :param skip_animation:
    :param key_debug_gene_connections:
    :return:
    """
    # Summary output of metrics
    metrics = []

    writer_pop_prefix = "{}-{} "  # prefix to add to tensorboard summarywriter labels
    writer_category = "Train"
    img_mode_prefix = ""
    if mode == "validate":
        writer_category = "Validation"
        img_mode_prefix = "val"
        g = "val"
    elif mode == "test":
        writer_category = "Test"
        img_mode_prefix = "Test"

    if not skip_animation and mode == "train":
        if sim_type == "2D":
            visualizer.generate_2D_anim(gen_no=g,
                                        output_path=path_config.anim_preview_dir + "/sim_anim_{}.mp4".format(g),
                                        fps=15, extra_text="Fitness: avg {:0.2f}, max {:0.2f}".format(
                    np.mean(populations[0].fitness),
                    populations[0].fitness.max()))
        elif sim_type == "numerical":
            # Generate bar plot animation
            visualizer.numerical_position_bar_animation(data_charts=env.data_out,
                                                        position_labels=["BTC", "ETH", "LTC", "USD"],
                                                        no_bars=20,
                                                        save_path=path_config.anim_preview_dir + "/sim_anim_{}.mp4".format(
                                                            g))
    # If numerical simulation, generate ROI plot preview and activity log
    if sim_type == "numerical":
        for p in range(len(populations)):
            if mode == "validate" or mode == "test":
                # Calculate the BAH population performance histogram
                bah_comparison, bah_compare_metric = visualizer.numerical_BAH_comparison(data_charts=env.data_out,
                                                                                         population_idx=p,
                                                                                         save_path="tensorboard")
                writer.add_scalar(
                    writer_category + "/" + writer_pop_prefix.format(p, populations[p].pop_label) + "ROI|ROI_BAH Mean",
                    bah_compare_metric['ROI/BAH_ROI Mean'], 0 if mode == "validate" else g)
                writer.add_histogram(
                    tag=writer_category + "/" + writer_pop_prefix.format(p, populations[p].pop_label) + "ROI|ROI_BAH",
                    values=bah_comparison, global_step=0 if mode == "validate" else g)
                metrics.append(bah_compare_metric)

    return metrics


def normalize_positions(positions):
    # Precalculate the ROI performance of populations and buy-and-hold
    positions = np.array(positions)

    if len(positions.shape) == 3:  # single population
        T, no_org, no_pos = positions.shape
        # before normalizing, if any position states are all 0s, set the last position index to 1
        mask = np.all(positions == 0, axis=-1)
        positions[mask, -1] = 1
        # normalize positions (to add =1 for each organism)
        positions = positions / np.tile(np.expand_dims(np.sum(positions, 2), 2), (1, 1, no_pos))
    else:  # Multiple populations
        no_pop, T, no_org, no_pos = positions.shape
        # before normalizing, if any position states are all 0s, set the last position index to 1
        mask = np.all(positions == 0, axis=-1)
        positions[mask, -1] = 1
        # normalize positions (to add up to 1.0 for each organism)
        positions = positions / np.tile(np.expand_dims(np.sum(positions, 3), 3), (1, 1, 1, no_pos))
    return positions


if __name__ == "__main__":
    # test numerical visualization
    np.random.seed(1)
    N_org = 1000
    T = 200
    pos = np.random.rand(T, N_org, 4)
    fitness = []
    for i in range(N_org):
        for t in range(T):
            pos[t, i, :] = pos[t, i, :] / np.sum(pos[t, i, :])
        fitness.append(i)
    fitness = np.array(fitness)
    charts = []
    charts.append(np.sin(np.linspace(0, 1, T + 10) * 20) * 0.3 + 1.)
    charts.append(np.sin(np.linspace(0, 1, T + 10) * 20 + 0.5) * 0.3 + 1.)
    charts.append(np.sin(np.linspace(0, 1, T + 10) * 20 + 1.5) * 0.3 + 1.)
    charts.append(np.sin(np.linspace(0, 1, T + 10) * 20 + 2.5) * 0.3 + 1.)
    charts = np.array(charts)

    visualizer = Visualizer()
    visualizer.fitness[0] = fitness
    visualizer.plot_pos[0] = pos

    '''    visualizer.numerical_position_bar_animation(data_charts=charts,
                                       position_labels=["BTC", "ETH", "LTC", "USD"], no_bars=20,
                                       save_path="./test.mp4")'''

    visualizer.numerical_position_roi_preview(data_charts=charts, plot_organism_idx=[0, 1],
                                              position_labels=["BTC", "ETH", "LTC", "USD"],
                                              save_path="./roi_preview.png")
