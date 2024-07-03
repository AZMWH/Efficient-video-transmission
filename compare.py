# coding=gbk
import ffmpy3
import subprocess

# 输入视频文件路径
input_file = 'compressed_video.avi'

# 创建 FFmpeg 对象
ff = ffmpy3.FFmpeg(executable=r'E:\study\YJS_work\1\ffmpeg-6.1.1-full_build\ffmpeg-6.1.1-full_build\bin\ffmpeg.exe',
    inputs={input_file: None},
    outputs={'-': ['-vf', 'showinfo', '-select_streams', 'v:0', '-show_entries', 'frame=pict_type', '-of', 'default=noprint_wrappers=1:nokey=1']}
)

# 运行命令
stdout, stderr = ff.run(stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# 将输出解码为字符串并按行拆分
stdout_lines = stdout.decode().split('\n')

# 输出第一帧的类型
print("第一帧的类型：", stdout_lines[0].strip())

# 输出第二帧的类型
print("第二帧的类型：", stdout_lines[1].strip())