import numpy as np
import scipy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def logistic_4_fitting(x, y):
    def func(x, b0, b1, b2, b3):
        return b1 + np.divide(b0 - b1, 1 + np.exp(np.divide(b2 - x, np.abs(b3))))
    x_axis = np.linspace(np.amin(x), np.amax(x), 100)
    init = np.array([np.max(y), np.min(y), np.mean(x), 0.1])
    popt, _ = curve_fit(func, x, y, p0=init, maxfev=int(1e8))
    curve = func(x_axis, *popt)
    fitted = func(x, *popt)

    return x_axis, curve, fitted


if __name__ == '__main__':
    import pandas as pd
    from PIL import Image
    import numpy as np
    from torchvision import transforms
    from tqdm import tqdm

    transform = transforms.ToTensor()
    df = pd.read_csv('/home/shanziyu/PAME/data/index/LS-PCQA/total_6view_512.csv')
    df_np = df
    for i,_ in tqdm(enumerate(range(len(df))),total=len(df),leave=False):
        img_path = df.iloc[i,1]
        df_np.iloc[i,1] = df_np.iloc[i,1].replace('proj_6view_1angle_512','proj_6view_1angle_512_np')
        df_np.iloc[i,1] = df_np.iloc[i,1].replace('.png','.npy')
        img = Image.open(img_path).convert('RGB')
        img = np.asarray(img)
        # np.save(df_np.iloc[i,1],img)
    df_np.to_csv('/home/shanziyu/PAME/data/index/LS-PCQA/total_6view_512_np.csv',index=False)

