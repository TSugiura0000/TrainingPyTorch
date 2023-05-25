from pdf2image import convert_from_path

# PDFファイルを読み込み、画像リストとして取得
images = convert_from_path('result/mnist_conv_nn_sgd.pdf')

for i, image in enumerate(images):
    # 画像をPNGフォーマットで保存
    image.save('result/mnist_conv_nn_sgd{}.png'.format(i), 'PNG')
