import numpy as np

def unfold(x, kh, kw, stride=1, pad=0):
    """Convert (batch_size, n_channels, in_h, in_w) -> (batch_size*out_h*out_w, n_channels*kh*kw) for convolution.

    kh, kw: kernel height and width
    stride: step size for sliding window
    pad: number of zeros added outside of the window
    """
    batch_size, n_channels, in_h, in_w = x.shape
    out_h = (in_h + 2 * pad - kh) // stride + 1
    out_w = (in_w + 2 * pad - kw) // stride + 1

    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    cols = np.zeros((batch_size, n_channels, kh, kw, out_h, out_w))

    # Extract the sliding window
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j, :, :] = xp[:, :, i:i + stride*out_h:stride,
                                        j:j + stride*out_w:stride]

    # (batch_size, n_channels, kh, kw, out_h, out_w) -> (batch_size, out_h, out_w, n_channels, kh, kw)
    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(batch_size * out_h * out_w, -1)

def fold(cols, x_shape, kh, kw, stride=1, pad=0):
    """Convert (batch_size*out_h*out_w, n_channels*kh*kw) -> (batch_size, n_channels, in_h, in_w) for convolution.

    kh, kw: kernel height and width
    stride: step size for sliding window
    pad: number of zeros added outside of the window
    """
    batch_size, n_channels, in_h, in_w = x_shape
    out_h = (in_h + 2 * pad - kh) // stride + 1
    out_w = (in_w + 2 * pad - kw) // stride + 1

    # (batch_size*out_h*out_w, n_channels*kh*kw) -> (batch_size, out_h, out_w, n_channels, kh, kw)
    cols = cols.reshape(batch_size, out_h, out_w, n_channels, kh, kw).transpose(0, 3, 4, 5, 1, 2)

    # Add the sliding window back to the image
    xp = np.zeros((batch_size, n_channels, in_h + 2 * pad, in_w + 2 * pad))
    for i in range(kh):
        for j in range(kw):
            xp[:, :, i:i + stride * out_h:stride,
               j:j + stride*out_w:stride] += cols[:, :, i, j, :, :]

    return xp[:, :, pad:pad+in_h, pad:pad+in_w]
