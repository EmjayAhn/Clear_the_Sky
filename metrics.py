import tensorflow as tf

def readfile_tensor(img_path):
    img_raw = tf.io.read_file(img_path)
    img_tensor = tf.image.decode_image(img_raw)
    return img_tensor


def ssim(img1_path, img2_path):
    img1_tensor = readfile_tensor(img1_path)
    img2_tensor = readfile_tensor(img2_path)
    
    max_elem = tf.math.reduce_max(img1_tensor)
    result_ssim = tf.image.ssim(img1_tensor, img2_tensor, max_val=max_elem)
    with tf.compat.v1.Session() as sess:
        result = sess.run(result_ssim)
        return result

def psnr(img1_path, img2_path):
    img1_tensor = readfile_tensor(img1_path)
    img2_tensor = readfile_tensor(img2_path)
    
    max_elem = tf.math.reduce_max(img1_tensor)
    result_psnr = tf.image.psnr(img1_tensor, img2_tensor, max_val=max_elem)
    with tf.compat.v1.Session() as sess:
        result = sess.run(result_psnr)
        return result