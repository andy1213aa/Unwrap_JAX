import pyzed.sl as sl
import cv2

def main():
    zed = sl.Camera()

    # 初始化ZED相机
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = 30


    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"相机打开失败：{status}")
        return

    runtime = sl.RuntimeParameters()

    # 获取相机信息
    camera_info = zed.get_camera_information()
    image_width = camera_info.camera_configuration.resolution.width
    image_height = camera_info.camera_configuration.resolution.height

    # 创建视频编码器
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    output_filename = "left_view.mp4"
    output_video = cv2.VideoWriter(output_filename, codec, 30, (image_width, image_height))

    # 开始录制视频
    while True:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            # 获取左视角图像
            left_image = sl.Mat()
            zed.retrieve_image(left_image, sl.VIEW.LEFT)

            # 转换为OpenCV格式
            left_frame = left_image.get_data()

            # 将图像写入视频文件
            output_video.write(left_frame)

            cv2.imshow("Left View", left_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    output_video.release()
    # 释放资源
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
