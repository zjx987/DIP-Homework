import cv2
import numpy as np
import gradio as gr
from scipy.spatial import Delaunay
from scipy.interpolate import Rbf
# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image


def thin_plate_spline_rbf(x, y):
    """Thin Plate Spline Radial Basis Function."""
    r = np.sqrt(x**2 + y**2)  # 计算点到原点的距离
    return np.where(r == 0, 0, r**2 * np.log(r))  # 计算薄板样条函数值

def gaussian_rbf(x, y, epsilon):
    """Gaussian Radial Basis Function."""
    return np.exp(-(epsilon * np.sqrt(x**2 + y**2))**2)
# 执行仿射变换

def point_guided_deformation(image, source_pts, target_pts, eps=0.01):
    """ 
    Return
    ------
        A deformed image.
    """

    height, width = image.shape[:2]
    epsilon= eps

    # 将源点和目标点转为 numpy 数组
    source_pts = np.array(source_pts)
    target_pts = np.array(target_pts)

    N = len(source_pts)
    # 创建一列全为 1 的数组
    ones_column = np.ones((target_pts.shape[0], 1))
    alpha= 0
    # 将列合并
    S = np.hstack((ones_column, target_pts))

    a = np.zeros(N+3).reshape(-1, 1)

    b = np.zeros(N+3).reshape(-1, 1)


# 创建一个包含三个零元素的数组
    zeros = np.zeros(3)  # shape (3,)

# 合并第一列和零元素
    u = np.concatenate((source_pts[:, 0], zeros)).reshape(-1, 1) # shape (n + 3,)
    v = np.concatenate((source_pts[:, 1], zeros)).reshape(-1, 1)
    G = np.zeros((N, N))

    for i in range(N):
         for j in range(N):
            G[i, j] = thin_plate_spline_rbf((target_pts[i] - target_pts[j])[0], (target_pts[i] - target_pts[j])[1])
            if j==i:
             G[i, j]= G[i, j]+alpha
            


    M = np.zeros((N + 3, N + 3))
    M[:N, :N] = G              # 前 N 行前 N 列为 G
    M[:N, N:N + 3] = S         # 前 N 行 N+1 到 N+3 列为 S
    M[N:N + 3, :N] = S.T       # N+1 到 N+3 行前 N 列为 S 的转置
    M[N:N + 3, N:N + 3] = np.zeros((3, 3)) 
    #a = np.dot(np.linalg.inv(np.dot(M.T,M)),np.dot(M.T,u))
    #b = np.dot(np.linalg.inv(np.dot(M.T,M)),np.dot(M.T,v))  
    a= np.linalg.solve(M,u)
    b= np.linalg.solve(M,v)
    ##M_inv = np.linalg.inv(M)

    #a = np.dot(M_inv, u)
    #b = np.dot(M_inv, v)
    x = np.zeros((height, width))
    y= np.zeros((height, width))
    warped_image = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            for k in range(N):
                x[i,j]=x[i,j]+a[k]*thin_plate_spline_rbf(i-target_pts[k, 1],j-target_pts[k, 0])
                y[i,j]=y[i,j]+b[k]*thin_plate_spline_rbf(i-target_pts[k, 1],j-target_pts[k, 0])
            x[i,j]=x[i,j]+a[N]+j*a[N+1]+i*a[N+2]
            y[i,j]=y[i,j]+b[N]+j*b[N+1]+i*b[N+2]
            x[i,j]=x[i,j].astype(int)
            y[i,j]=y[i,j].astype(int)
            if x[i,j] >width -1:
             x[i,j]=width-1
            if y[i,j] > height-1:
             y[i,j]= height-1
            if x[i,j] < 0:
             x[i,j]=0
            if y[i,j] < 0:
             y[i,j]=0           
            warped_image[i, j] = image[int(y[i, j]),int(x[i, j])]
    ### FILL: 基于MLS or RBF 实现 image warping

    return warped_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch(share=True)
