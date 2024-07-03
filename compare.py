# coding=gbk
import ffmpy3
import subprocess

# ������Ƶ�ļ�·��
input_file = 'compressed_video.avi'

# ���� FFmpeg ����
ff = ffmpy3.FFmpeg(executable=r'E:\study\YJS_work\1\ffmpeg-6.1.1-full_build\ffmpeg-6.1.1-full_build\bin\ffmpeg.exe',
    inputs={input_file: None},
    outputs={'-': ['-vf', 'showinfo', '-select_streams', 'v:0', '-show_entries', 'frame=pict_type', '-of', 'default=noprint_wrappers=1:nokey=1']}
)

# ��������
stdout, stderr = ff.run(stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# ���������Ϊ�ַ��������в��
stdout_lines = stdout.decode().split('\n')

# �����һ֡������
print("��һ֡�����ͣ�", stdout_lines[0].strip())

# ����ڶ�֡������
print("�ڶ�֡�����ͣ�", stdout_lines[1].strip())