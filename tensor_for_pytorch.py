import torch


def gray_image():
    img_t = torch.ones(3, 5, 5)  # shape[channels, rows, columns]
    print('img_t = ', img_t, '\n')
    weights = torch.tensor([0.2126, 0.7152, 0.0722])

    batch_t = torch.ones(2, 3, 5, 5)  # shape[batch, channels, rows, columns]
    print('batch_t', batch_t, '\n')

    unsqueezed_weights = weights.unsqueeze(-1).unsqueeze(-1)
    print('unsqueezed_weights = ', unsqueezed_weights, '\n')

    img_weights = (img_t * unsqueezed_weights)
    print('img_weights = ', img_weights, '\n')

    batch_weights = (batch_t * unsqueezed_weights)
    print('batch_weights = ', batch_weights, '\n')

    img_gray_weighted = img_weights.sum(-3)
    print('img_gray_weighted', img_gray_weighted, '\n')

    batch_gray_weighted = batch_weights.sum(-3)
    print('batch_gray_weighted', batch_gray_weighted, '\n')


def test_tensor_product():
    test_t = torch.ones(4, 2) * 10
    print('test_t = \n', test_t, '\n')

    weight_v = torch.tensor([1, 2, 3, 4]).unsqueeze(-1)
    print('weight_v = \n', weight_v, '\n')
    weighted_test_t = test_t * weight_v
    print('weighted_test_t = \n', weighted_test_t, '\n')

    weight_t = torch.tensor(
        [[[1, 5], [2, 6], [3, 7], [4, 8]],
         [[9, 13], [10, 14], [11, 15], [12, 16]]]
    )
    print('weight_t.shape = ', weight_t.shape)
    print('weight_t = \n', weight_t)

    weighted_test_t = test_t * weight_t
    print('weighted_test_t = \n', weighted_test_t, '\n')


def torch_functions():
    b = torch.ones(3, 4)
    print(b)

    c = torch.rand(4, 5, 2)
    print(c)


def indexing():
    d = torch.randint(-5, 6, (5, 3))
    d1 = d[1:]      # 行1以降を指定，列も全部指定
    d2 = d[2:, :]   # 行2以降を指定，列は全部指定
    d3 = d[3:, 1]   # 行3以降を指定，列1を指定
    d4 = d[None]    # 新たな次元を追加．unsqueezeと同じ

    print('d = \n{0}\n{1}\n'.format(d, d.shape))
    print('d[1:] = \n{0}\n{1}\n'.format(d1, d1.shape))
    print('d[2:, :] = \n{0}\n{1}\n'.format(d2, d2.shape))
    print('d[3:, 1] = \n{0}\n{1}\n'.format(d3, d3.shape))
    print('d[None] = \n{0}\n{1}\n'.format(d4, d4.shape))


def sum():
    e = torch.randn(3, 5, 5)
    e_sum = e.sum(-3)
    print('e = \n{0}\n{1}'.format(e, e.shape))
    print('e_sum = \n{0}\n{1}'.format(e_sum, e_sum.shape))


def calc():
    a = torch.randint(0, 5, (3, 3))
    b = torch.randint(0, 5, (3, 3))
    c = a + b
    d = a - b
    e = a * b
    f = a / b
    print(f'a = \n{a}\n{a.shape}')
    print(f'b = \n{b}\n{b.shape}')
    print(f'c = \n{c}\n{c.shape}')
    print(f'd = \n{d}\n{d.shape}')
    print(f'e = \n{e}\n{e.shape}')
    print(f'f = \n{f}\n{f.shape}')


def gray_scale():
    img = torch.randn(3, 5, 5)
    weight = torch.tensor([0.2126, 0.7152, 0.0722])
    unsqueezed_weight = weight.unsqueeze(-1).unsqueeze(-1)
    weighted_img = img * unsqueezed_weight
    gray_img = weighted_img.sum(-3)
    print(img.shape)
    print(weight.shape)
    print(unsqueezed_weight.shape)
    print(weighted_img.shape)
    print(gray_img.shape)


def meta_info():
    points = torch.randn(3, 2)
    third_point = points[2]
    print(third_point.storage_offset())
    print(third_point.size())
    print(third_point.stride())


if __name__ == '__main__':
    meta_info()
