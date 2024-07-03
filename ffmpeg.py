#
import ffmpy3


# 输入视频文件名和输出文件名
def extract_frames(frame_room, xyxy_video):
    # 时域切割重要区域
    input_video = 'video/text/text2.avi'
    output_video = 'video/output/text_QG.avi'
    start_frame = frame_room[0]
    end_frame = frame_room[-1]

    ffmpeg_command = ffmpy3.FFmpeg(
        inputs={input_video: None},
        outputs={output_video: ['-vf', 'select=between(n\,{0}\,{1})'.format(start_frame, end_frame), '-y']}
    )
    # 执行 FFmpeg 命令
    ffmpeg_command.run()

    output_video2 = 'video/output/text_QG2.avi'
    ffmpeg_command = ffmpy3.FFmpeg(
        inputs={output_video: None},
        outputs={output_video2: ['-vf', 'mpdecimate,setpts=N/FRAME_RATE/TB', '-y']}
    )
    ffmpeg_command.run()

    # 横切重要区域
    output_file1 = 'video/test/output_video1.avi'
    output_file2 = 'video/test/output_video2.avi'
    output_file3 = 'video/test/output_video3.avi'
    x1, y1, x2, y2 = xyxy_video
    w = x2 - x1
    h = y2 - y1
    # 构建crop参数字符串
    crop_params = ['-vf', 'crop={}:{}:{:d}:{:d}'.format(w, h, x1, y1), '-y']
    # 创建FFmpeg对象并执行裁剪命令
    ffmpeg = ffmpy3.FFmpeg(
        inputs={output_video2: None},
        outputs={output_file1: crop_params}
    )
    ffmpeg.run()
    # 背景区域1
    w = 640
    h = y1
    x3 = 0
    y3 = 0
    # 构建crop参数字符串
    crop_params = ['-vf', 'crop={}:{}:{:d}:{:d}'.format(w, h, x3, y3), '-y']

    # 创建FFmpeg对象并执行裁剪命令
    ffmpeg = ffmpy3.FFmpeg(
        inputs={output_video2: None},
        outputs={output_file2: crop_params}
    )
    ffmpeg.run()
    # 背景区域2
    w = 640 - x2
    h = y2 - y1
    x4 = x2
    y4 = y1
    # 构建crop参数字符串
    crop_params = ['-vf', 'crop={}:{}:{:d}:{:d}'.format(w, h, x4, y4), '-y']

    # 创建FFmpeg对象并执行裁剪命令
    ffmpeg = ffmpy3.FFmpeg(
        inputs={output_video2: None},
        outputs={output_file3: crop_params}
    )
    ffmpeg.run()

    # 压缩重要区域视频
    input_vid = output_file1
    out_vid1 = 'video/output/compression_important.avi'

    compression_params = [
        '-c:v', 'libx264',  # 使用H.264视频编解码器
        '-crf', '30',  # 设置视频质量，CRF值为45
        '-preset', 'medium',  # 设置编码速度/质量的平衡，默认为medium
        '-r', '15',
        '-y'
    ]

    # 创建FFmpeg对象并执行压缩命令
    ffmpeg = ffmpy3.FFmpeg(
        inputs={input_vid: None},
        outputs={out_vid1: compression_params}
    )
    ffmpeg.run()
    # 压缩背景1
    input_vid = output_file2
    out_vid2 = 'video/output/compression_bg1.avi'

    compression_params = [
        '-c:v', 'libx264',  # 使用H.264视频编解码器
        '-crf', '30',  # 设置视频质量，CRF值为45
        '-preset', 'medium',  # 设置编码速度/质量的平衡，默认为medium
        '-r', '15',
        '-y'
    ]

    # 创建FFmpeg对象并执行压缩命令
    ffmpeg = ffmpy3.FFmpeg(
        inputs={input_vid: None},
        outputs={out_vid2: compression_params}
    )
    ffmpeg.run()
    # 压缩背景2
    input_vid = output_file3
    out_vid3 = 'video/output/compression_bg2.avi'

    compression_params = [
        '-c:v', 'libx264',  # 使用H.264视频编解码器
        '-crf', '30',  # 设置视频质量，CRF值为45
        '-preset', 'medium',  # 设置编码速度/质量的平衡，默认为medium
        '-r', '15',
        '-y'
    ]

    # 创建FFmpeg对象并执行压缩命令
    ffmpeg = ffmpy3.FFmpeg(
        inputs={input_vid: None},
        outputs={out_vid3: compression_params}
    )
    ffmpeg.run()

    # 提取重要区域的IPB帧
    input_file = out_vid1

    output_i_frames = 'video/output/output_important/i_frames.avi'
    output_p_frames = 'video/output/output_important/p_frames.avi'
    output_b_frames = 'video/output/output_important/b_frames.avi'

    # FFmpeg命令提取I帧
    ff_i_frames = ffmpy3.FFmpeg(
        inputs={input_file: None},
        outputs={output_i_frames: ['-vf', 'select=\'eq(pict_type,I)\'', '-vsync', '0', '-y']}
    )
    ff_i_frames.run()

    # FFmpeg命令提取P帧
    ff_p_frames = ffmpy3.FFmpeg(
        inputs={input_file: None},
        outputs={output_p_frames: ['-vf', 'select=\'eq(pict_type,P)\'', '-vsync', '0', '-y']}
    )
    ff_p_frames.run()

    # FFmpeg命令提取B帧
    ff_b_frames = ffmpy3.FFmpeg(
        inputs={input_file: None},
        outputs={output_b_frames: ['-vf', 'select=\'eq(pict_type,B)\'', '-vsync', '0', '-y']}
    )
    ff_b_frames.run()
    print('已完成')
    # 提取bj1的IPB帧
    input_file = out_vid2

    output_i_frames = 'video/output/output_bj1/i_frames.avi'
    output_p_frames = 'video/output/output_bj1/p_frames.avi'
    output_b_frames = 'video/output/output_bj1/b_frames.avi'

    # FFmpeg命令提取I帧
    ff_i_frames = ffmpy3.FFmpeg(
        inputs={input_file: None},
        outputs={output_i_frames: ['-vf', 'select=\'eq(pict_type,I)\'', '-vsync', '0', '-y']}
    )
    ff_i_frames.run()

    # FFmpeg命令提取P帧
    ff_p_frames = ffmpy3.FFmpeg(
        inputs={input_file: None},
        outputs={output_p_frames: ['-vf', 'select=\'eq(pict_type,P)\'', '-vsync', '0', '-y']}
    )
    ff_p_frames.run()

    # FFmpeg命令提取B帧
    ff_b_frames = ffmpy3.FFmpeg(
        inputs={input_file: None},
        outputs={output_b_frames: ['-vf', 'select=\'eq(pict_type,B)\'', '-vsync', '0', '-y']}
    )
    ff_b_frames.run()

    # 提取bj2的IPB帧
    input_file = out_vid3

    output_i_frames = 'video/output/output_bj2/i_frames.avi'
    output_p_frames = 'video/output/output_bj2/p_frames.avi'
    output_b_frames = 'video/output/output_bj2/b_frames.avi'

    # FFmpeg命令提取I帧
    ff_i_frames = ffmpy3.FFmpeg(
        inputs={input_file: None},
        outputs={output_i_frames: ['-vf', 'select=\'eq(pict_type,I)\'', '-vsync', '0', '-y']}
    )
    ff_i_frames.run()

    # FFmpeg命令提取P帧
    ff_p_frames = ffmpy3.FFmpeg(
        inputs={input_file: None},
        outputs={output_p_frames: ['-vf', 'select=\'eq(pict_type,P)\'', '-vsync', '0', '-y']}
    )
    ff_p_frames.run()

    # FFmpeg命令提取B帧
    ff_b_frames = ffmpy3.FFmpeg(
        inputs={input_file: None},
        outputs={output_b_frames: ['-vf', 'select=\'eq(pict_type,B)\'', '-vsync', '0', '-y']}
    )
    ff_b_frames.run()
