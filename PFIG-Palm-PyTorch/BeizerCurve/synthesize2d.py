import bezier, mmcv
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

from multiprocessing import Pool
import os, argparse, cv2, random
from os.path import join, dirname
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def parse_args():
    parser = argparse.ArgumentParser(description='Compare results')
    parser.add_argument('--num_ids', type=int, default=100)
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--nproc', type=int, default=5)
    parser.add_argument('--imsize', type=int, default=256)
    parser.add_argument('--output', type=str, default='./line')
    args = parser.parse_args()
    assert args.num_ids % args.nproc == 0
    return args


def wrap_points(points, M):
    assert isinstance(points, np.ndarray)
    assert isinstance(M, np.ndarray)
    n = points.shape[0]
    augmented_points = np.concatenate((points, np.ones((n, 1))), axis=1).astype(points.dtype)
    points = (M @ augmented_points.T).T
    points = points / points[:,-1].reshape(-1, 1)
    return points[:, :2]


def sample_edge(low, high):
    """
    sample points on edges of a unit square
    """
    offset = min(low, high)
    low, high = map(lambda x: x - offset, [low, high])
    t = np.random.uniform(low, high) + offset

    if t >= 4:
        t = t % 4
    if t < 0:
        t = t + 4

    if t <= 1:
        x, y = t, 0
    elif 1 < t <= 2:
        x, y = 1, t - 1
    elif 2 < t <= 3:
        x, y = 3 - t, 1
    else:
        x, y = 0, 4 - t
    return np.array([x, y]), t

def control_point(head, tail, t=0.5, s=0):
    head = np.array(head)
    tail = np.array(tail)
    l = np.sqrt(((head - tail) ** 2).sum())
    assert head.size == 2 and tail.size == 2
    assert l >= 0
    c = head * t + (1 - t) * tail
    x, y = head - tail
    v = np.array([-y, x])
    v /= max(np.sqrt((v ** 2).sum()), 1e-6)
    return c + s * l * v


def get_bezier(p0, p1, t=0.5, s=1):
    assert -1 < s < 1, 's=%f'%s
    c = control_point(p0, p1, t, s)
    nodes = np.vstack((p0, c, p1)).T
    return bezier.Curve(nodes, degree=2)

def generate_parameters():
    # head coordinates
    head1, _ = sample_edge(0, 0.4)
    head2, _ = sample_edge(-0.5, -0.4 + 0.5*head1[0])
    head3, _ = sample_edge(-head2[1], -head2[1]-0.1)
    head4, _ = sample_edge(0.2, 0.4)

    # tail coordinates
    tail1, _ = sample_edge(1.2, 1.6)
    tail2, _ = sample_edge(tail1[1]+1.3, 2.1)
    tail3, _ = sample_edge(2.35, 2.8)
    tail4, _ = sample_edge(2.2, 2.4)

    # control coordinates
    c1 = control_point(head1, tail1, s = -np.random.uniform(0.1 + head1[1]/6, 0.15 + head1[1]/6))
    c2 = control_point(head2, tail2, s = np.random.uniform(0.1, 0.15))
    c3 = control_point(head3, tail3, s = np.random.uniform(0.15, 0.25))
    c4 = control_point(head4, tail4, s = np.random.uniform(-0.01, 0.01))

    return np.vstack((head1, c1, tail1)), np.vstack((head2, c2, tail2)), np.vstack((head3, c3, tail3)), np.vstack((head4, c4, tail4))


def batch_process(proc_index, ranges, args):
    ids_per_proc = int(args.num_ids / args.nproc)
    EPS = 1e-2

    np.random.seed(proc_index)
    random.seed(proc_index)
    local_idx = 0
    for id_idx, i in enumerate(range(*ranges[proc_index])):
        # start/end points of main creases
        nodes1 = generate_parameters()
        start1 = [np.random.uniform(0, 0.25), np.random.uniform(0, 0.1),
                  np.random.uniform(0, 0.15), np.random.uniform(0, 0.25)]
        end1 = [np.random.uniform(1, 1), np.random.uniform(0.75, 1),
                  np.random.uniform(0.9, 1), np.random.uniform(0.75, 1)]

        flag1 = [np.random.uniform()>0.005, np.random.uniform()>0.005, np.random.uniform()>0.005, np.random.uniform()>0.9]

        # start/end points of secondary creases
        n2 = np.random.randint(8, 15)

        mean_distance = 50
        var_distance = 12
        coord2 = []
        for _ in range(n2):
            start_point = np.random.uniform(0, args.imsize, size=(2,))
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.normal(mean_distance, var_distance)
            end_point_x = start_point[0] + np.cos(angle) * distance
            end_point_y = start_point[1] + np.sin(angle) * distance

            end_point = np.array([np.clip(end_point_x, 0, args.imsize), np.clip(end_point_y, 0, args.imsize)])

            coord2.append([start_point, end_point])

        coord2 = np.array(coord2)

        s2 = np.clip(np.random.normal(scale=0.3, size=(n2,)), -0.4, 0.4) 
        t2 = np.clip(np.random.normal(loc=0.5, scale=0.4, size=(n2,)), 0.3, 0.7)

        # synthesize samples for each ID
        for s in range(args.samples):
            fig = plt.figure(frameon=False)
            canvas = fig.canvas
            dpi = fig.get_dpi()
            fig.set_size_inches((args.imsize + EPS) / dpi, (args.imsize + EPS) / dpi)  
            # remove white edges by set subplot margin


            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)  
            ax = plt.gca()
            ax.set_xlim(0, args.imsize)
            ax.set_ylim(args.imsize, 0)
            ax.axis('off')

            bg_im = np.ones((args.imsize, args.imsize, 3), dtype=np.uint8) * 255
            bg_id = -1
            bg = {'filename': 'none'}


            bg_im = Image.fromarray(bg_im)
            ax.imshow(bg_im)

            # main creases
            curves1 = []
            for index, n in enumerate(nodes1):
                if index == 0:
                    curves = bezier.Curve(n.T * args.imsize + np.random.uniform(-10, 10, size=n.T.shape), degree=2)
                elif index == 2:
                    curves = bezier.Curve(n.T * args.imsize + np.random.uniform(-10, 10, size=n.T.shape), degree=2)
                else:
                    curves = bezier.Curve(n.T * args.imsize + np.random.uniform(-10, 10, size=n.T.shape), degree=2)
                curves1.append(curves)

            points1 = [c.evaluate_multi(np.linspace(s, e, 50)).T for c, s, e in zip(curves1, start1, end1)]


            paths1 = [Path(p) for p in points1]
            lw1 = np.random.uniform(3.6, 4.1, size=3)
            lw1 = np.append(lw1, np.random.uniform(1.6, 2.1))
            patches1 = [patches.PathPatch(p, edgecolor=np.random.uniform(0, 0.0, 3), facecolor='none', lw=lw1[i]) for i, p in enumerate(paths1)]
            for p, f in zip(patches1, flag1):
                if f:
                    ax.add_patch(p)

            # secondary creases
            # add turbulence to each sample
            coord2_ = coord2 + np.random.uniform(-5, 5, coord2.shape)
            s2_ = s2 + np.random.uniform(-0.1, 0.1, s2.shape)
            t2_ = t2 + np.random.uniform(-0.05, 0.05, s2.shape)

            lw2 = np.random.uniform(1.3, 1.5)
            for j in range(n2):
                points2 = get_bezier(coord2_[j, 0], coord2_[j, 1], t=t2_[j], s=s2_[j]).evaluate_multi(np.linspace(0, 1, 50)).T
                p = patches.PathPatch(Path(points2), edgecolor=np.random.uniform(0, 0.0, 3), facecolor='none', lw=lw2)
                ax.add_patch(p)

            stream, _ = canvas.print_to_buffer()
            buffer = np.frombuffer(stream, dtype='uint8')
            img_rgba = buffer.reshape(args.imsize, args.imsize, 4)
            rgb, alpha = np.split(img_rgba, [3], axis=2)
            img = rgb.astype('uint8')
            img = mmcv.rgb2bgr(img)

            if i >= args.num_ids/2:
                img = cv2.flip(img, 1)

            filename = join(args.output, '%.5d' % i, '%.3d.jpg' % s)
            os.makedirs(dirname(filename), exist_ok=True)
            mmcv.imwrite(img, filename)
            plt.close()
            local_idx += 1

        print("proc[%.3d/%.3d] id=%.4d [%.4d/%.4d]  " % (proc_index, args.nproc, i, id_idx, ids_per_proc))


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    spacing = np.linspace(0, args.num_ids,  args.nproc + 1).astype(int)

    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    argins = []
    for p in range(args.nproc):
        argins.append([p, ranges, args])

    with Pool() as pool:
        pool.starmap(batch_process, argins)
