import torch
import numpy as np
import math

# input: model (not in Data Parallel), image in tensor, ...
# detect_window = tensor size that can fit in memory for one convolution
def detect_image(model, image, section_size=400):
    _, w, h = image.size()
    window_size = model.window_size
    num_class = model.num_class
    # result
    l, r = math.ceil(window_size / 2) - 1, math.floor(window_size / 2)  # left (up) and right (down) paddings
    res_w, res_h = w - window_size + 1, h - window_size + 1             # size of results
    res = torch.FloatTensor(num_class + 1, res_w, res_h).fill_(1)
    step = section_size - window_size + 1
    for i in range(0, res_w, step):
        for j in range(0, res_h, step):
            model.train(False)
            # top left = i, j
            x, y = min(res_w - 1, i + step - 1), min(res_h - 1, j + step - 1)
            cur_img = image[:, i:x+window_size, j:y+window_size].clone()
            cur_pred = model(cur_img.unsqueeze(0))
            res[:, i:x+1, j:y+1] = cur_pred
            torch.cuda.empty_cache()
            del cur_img
            del cur_pred
#             print('from img: ({}:{}, {}:{})'.format(i, x + window_size, j, y + window_size))
#             print('from res: ({}:{}, {}:{})'.format(i, x + 1, j, y + 1))
    return res


def sample_nonobj_from_image(labels, selected_labels=None, method='mixed'):
    m = method
    if m == 'mixed':
        m = ['hard', 'uniform'][np.random.choice(2, 1)[0]]

    count = 0
    while True:
        count += 1
        if count == 100:
            print('something went wrong...')
            return None
        if m == 'hard':
            min_r, max_r = 0.02, 0.07
            r = (max_r - min_r) * np.random.random() + min_r
            dt = 2 * np.pi * np.random.random()
            dx, dy = r * np.cos(dt), r * np.sin(dt)
            x, y = selected_labels[np.random.choice(selected_labels.shape[0], 1), [0, 1]]
            x += dx
            y += dy
        elif m == 'uniform':
            x, y = np.random.random(), np.random.random()
        else:
            return None

        if x < 0 or x > 1 or y < 0 or y > 1:
            continue

        dist = np.array([np.sqrt((x - p[0]) ** 2 + (y - p[1]) ** 2) for _, p in enumerate(labels)])
        if dist.min() > 0.02:
            return x, y


def evaluate_detection(df_labl, df_pred, path_range=None, thres=0.02):
    dis = []

    if path_range is None:
        path_range = df_labl['path'].unique()

    for path in path_range:
        # class [pos, neg]
        for c in [0, 1]:
            df_cur_labl = df_labl[(df_labl['path'] == path) & (df_labl['class'] == c)]
            df_cur_pred = df_pred[(df_pred['path'] == path) & (df_pred['class'] == c)]

            # [label index, prediction index, distance]
            cur_dis = np.array([[i, j, np.sqrt((l['x'] - p['x']) ** 2 + (l['y'] - p['y']) ** 2)]
                                for i, l in df_cur_labl.iterrows()
                                for j, p in df_cur_pred.iterrows()])

            visited_pred = []
            visited_labl = []
            for index in np.argsort(cur_dis[:, 2]):
                # break if current distance is greater than the threshold
                if cur_dis[index, 2] >= thres:
                    break

                cur_labl_idx = int(cur_dis[index, 0])
                cur_pred_idx = int(cur_dis[index, 1])

                if (cur_labl_idx not in visited_labl) and (cur_pred_idx not in visited_pred):
                    dis.append(cur_dis[index, 2])
                    visited_pred.append(cur_pred_idx)
                    visited_labl.append(cur_labl_idx)

                    df_pred.at[cur_pred_idx, 'detected'] = 1
                    df_labl.at[cur_labl_idx, 'detected'] = 1

    return np.mean(dis)

def get_stats(pred, labl, mode='total'):
    if mode == 'pos':
        pred = pred[pred['class'] == 0].copy()
        labl = labl[labl['class'] == 0].copy()
    elif mode == 'neg':
        pred = pred[pred['class'] == 1].copy()
        labl = labl[labl['class'] == 1].copy()

    total_pred = len(pred)
    total_labl = len(labl)
    correct_pred = len(pred[(pred['detected'] == 1)])
    correct_labl = len(labl[(labl['detected'] == 1)])
    recall = correct_labl / total_labl
    precision = correct_pred / total_pred

    print('mode = {}'.format(mode))
    print('recall = %.3f ' % recall)
    print('precision = %.3f' % precision)
    print('F1 score = %.3f' % (2 * recall * precision / (recall + precision)))

# def detect_image(model, image):
#     model.train(False)
#     window_size = model.window_size
#     num_class = model.num_class
#
#     pred = model(image)
#     pred = torch.softmax(pred, dim=-1)


# # given prediction result of one cell, assign every prediction with one labeled defect (if exists)
# # pred: [N * (num_classes + 3)] (obj, x, y, class prob...), labels = [M * 3 (x, y, class)]
# def build_target(pred, labels, dist_thres=0.2):
#     BoolTensor = torch.cuda.BoolTensor if pred.is_cuda else torch.BoolTensor
#     FloatTensor = torch.cuda.FloatTensor if pred.is_cuda else torch.FloatTensor
#     LongTensor = torch.cuda.LongTensor if pred.is_cuda else torch.LongTensor
#
#     N = pred.size(0)         # number of prediction
#     M = labels.size(0)       # number of labels
#     C = pred.size(1) - 3     # number of classes
#
#     dist = FloatTensor(N, M).fill_(0)  # distance
#     obj_mask = BoolTensor(N).fill_(0)
#     noobj_mask = BoolTensor(N).fill_(1)
#     class_mask = BoolTensor(N).fill_(0)
#     tx = FloatTensor(N).fill_(0)
#     ty = FloatTensor(N).fill_(0)
#     tcls = FloatTensor(N, C).fill_(0)
#
#     for i in range(N):
#         for j in range(M):
#             px, py = pred[i, 1], pred[i, 2]     # prediction
#             lx, ly = labels[j, 0], labels[j, 1]  # labels
#             dist[i, j] = torch.sqrt((px - lx)**2 + (py - ly)**2)
#     best_n = dist.argmin(dim=0) if dist.size(1) > 0 else LongTensor([])
#     best_n = best_n.detach()
#
#     obj_mask[best_n] = 1
#     noobj_mask[best_n] = 0
#
#     for col in range(M):
#         noobj_mask[dist[:, col] < dist_thres] = 0
#
#     tx[best_n] = labels[:, 0]
#     ty[best_n] = labels[:, 1]
#
#     class_label = labels[:, 2].long()
#     class_pred = pred[:, 3:]  # probability of every class predicted
#
#     tcls[best_n, class_label] = 1
#     # class_mask[best_n] = (class_pred.argmax(-1) == class_label).float()
#     tconf = obj_mask.float()
#     return class_mask, obj_mask, noobj_mask, tx, ty, tcls, tconf
#
#
# # normalize labeled positions in images to every cell
# # input: labels = [N, 3], grid_size = number of grid
# # output: grid label, [grid, grid]
# def label2grid_label(labels, grid_size):
#     FloatTensor = torch.cuda.FloatTensor if labels.is_cuda else torch.FloatTensor
#     # grid_label = np.zeros((grid_size, grid_size), dtype=object)
#     # grid_label.fill(np.empty([0, 3]))
#     grid_label = np.zeros((grid_size, grid_size), dtype=object)
#     grid_label.fill(FloatTensor(0, 3))
#     grid_label = grid_label.tolist()
#     for i in range(labels.shape[0]):
#         if torch.eq(labels[i, :], -1).all():
#             break
#         x, y, c = labels[i, :]
#         px, py = x // (1/grid_size), y // (1/grid_size)
#         px, py = min(px, grid_size-1), min(py, grid_size-1)
#         px, py = max(px, 0), max(py, 0)
#
#         px, py = int(px), int(py)
#
#         prev = grid_label[py][px]
#         grid_label[py][px] = torch.cat((prev, FloatTensor([[
#             (x % (1/grid_size)) / (1/grid_size), (y % (1/grid_size)) / (1/grid_size), c
#         ]])), dim=0)
#     return grid_label
#
#
# def grid_label2label(grid_label):
#     label = []
#     n = len(grid_label)
#     m = len(grid_label[0])
#     for i in range(n):
#         for j in range(m):
#             for p in grid_label[i][j]:
# #                 if np.all(p == -1):
# #                     break
#                 label.append([p[0]/n + j/n, p[1]/m + i/m, p[2]])
#     return np.array(label)
#
#
# # expected input size: (anchor_num, grid, grid, 3 + class_num)
# # output size: (N, 3 + class_num)
# def grid_pred2pred(grid_pred):
#     grid_pred = grid_pred.permute(1, 2, 0, 3)
#     N, M, A, C = grid_pred.size()
#
#     x_offset = torch.from_numpy(np.arange(M)).expand((M, M)).float()
#     y_offset = torch.from_numpy(np.arange(N)).unsqueeze(0).t().expand((N, N)).float()
#
#     # adjust x
#     grid_pred[:, :, :, 1] = grid_pred[:, :, :, 1] + x_offset.unsqueeze(-1).expand_as(grid_pred[:, :, :, 1])
#     grid_pred[:, :, :, 1] = grid_pred[:, :, :, 1] / M
#     # adjust y
#     grid_pred[:, :, :, 2] = grid_pred[:, :, :, 2] + y_offset.unsqueeze(-1).expand_as(grid_pred[:, :, :, 2])
#     grid_pred[:, :, :, 2] = grid_pred[:, :, :, 2] / N
#
#     grid_pred = grid_pred.contiguous()
#     grid_pred = grid_pred.view(N * M * A, C)
#
#     return grid_pred


# def non_max_suppression(pred, conf_thres=0.5, dist_thres=0.05):
#     output = [None] * pred.size(0)
#     for image_i, image_pred in enumerate(pred):
#         image_pred = image_pred.squeeze()
#         image_pred = grid_pred2pred(image_pred)
#         image_pred = image_pred[image_pred[:, 0] > conf_thres]
#         if not image_pred.size(0):
#             continue
#         score = image_pred[:, 0] * image_pred[:, 3:].max(1)[0]
#         image_pred = image_pred[(-score).argsort()]
#         cls_conf, cls_pred = image_pred[:, 3:].max(1, keepdim=True)
#         detection = torch.cat((image_pred[:, :3], cls_conf.float(), cls_pred.float()), dim=1)
#         keep_pts = []
#         while detection.size(0):
#             # distance from
#             px, py = detection[0, 1], detection[0, 2]
#             dis = [torch.sqrt((px - p[1]) ** 2 + (py - p[2]) ** 2) for _, p in enumerate(detection)]
#             dis = torch.FloatTensor(dis)
#             dis_mask = dis < dist_thres
#             lbl_mask = detection[0, -1] == detection[:, -1]
#             invalid = dis_mask & lbl_mask
#
#             keep_pts += [detection[0, 1:]]
#             detection = detection[~invalid]
#
#         if keep_pts:
#             output[image_i] = torch.stack(keep_pts)
#     return output
#
#
# def get_batch_stats(output, target, dist_thres=0.05):
#     total_pred = 0
#     total_labl = 0
#     true_positive = 0
#     assert(len(output) == len(target))
#     for sample_i in range(len(output)):
#         image_pred = output[sample_i]
#         image_labl = target[sample_i]
#         if image_pred is None:
#             continue
#         total_pred += image_pred.size(0)
#         total_labl += image_labl.size(0)
#         for target_i, t in enumerate(image_labl):
#             tx, ty = t[0], t[1]
#             dis = torch.FloatTensor([torch.sqrt((p[0] - tx)**2 + (p[1] - ty)**2) for _, p in enumerate(image_pred)])
#             dis_mask = dis.argsort()
#             cls_mask = image_pred[:, 2] == t[2]
#             mask = dis_mask * cls_mask
#             if dis[mask[0]] < dist_thres:
#                 true_positive += 1
#                 new_mask = torch.BoolTensor(image_pred.size(0)).fill_(1)
#                 new_mask[mask[0]] = 1
#                 image_pred = image_pred[new_mask]
#
#     return total_pred, total_labl, true_positive
