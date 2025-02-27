import json
import gzip
import os


def convert_episode_format(episode, episode_id):
    """转换单个episode的格式"""
    # 提取glb文件名
    scene_id = episode['scene_id']
    if '/' in scene_id:
        glb_name = scene_id.split('/')[-1]
    else:
        glb_name = scene_id

        # 处理goals，删除rotation并添加radius
    new_goals = []
    for goal in episode['goals']:
        new_goal = {
            'position': goal['position'],
            'radius': None
        }
        new_goals.append(new_goal)

        # 创建新的episode字典
    new_episode = {
        'episode_id': episode_id,
        'scene_id': f"data/scene_datasets/gibson/{glb_name}",
        'start_position': episode['start_position'],
        'start_rotation': episode['start_rotation'],
        'info': {
            'geodesic_distance': episode.get('length_shortest', 0),
            'difficulty': 'easy'
        },
        'goals': new_goals,
        'shortest_paths': None,
        'start_room': None
    }

    return new_episode


def convert_json_format(input_data):
    """转换整个JSON数据的格式"""
    # 确保输入数据是字典类型且包含episodes键
    if not isinstance(input_data, dict) or 'episodes' not in input_data:
        raise ValueError("输入数据格式错误，应该包含'episodes'键")

        # 转换每个episode
    new_episodes = []
    for i, episode in enumerate(input_data['episodes']):
        new_episode = convert_episode_format(episode, i)
        new_episodes.append(new_episode)

        # 返回新的数据结构
    return {'episodes': new_episodes}


def save_as_gzip(data, output_file):
    """将数据保存为gzip压缩的JSON文件"""
    json_str = json.dumps(data)
    json_bytes = json_str.encode('utf-8')
    with gzip.open(output_file, 'wb') as f:
        f.write(json_bytes)


def read_json_or_gzip(file_path):
    """读取JSON或JSON.GZ文件"""
    try:
        # 首先尝试作为gzip文件读取
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                return json.load(f)
                # 否则作为普通JSON文件读取
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"读取文件时发生错误：{str(e)}")
        raise


def main():
    # 文件名设置
    input_file = '/home/hsf/FGPrompt/data/datasets/imagenav/mp3d/v1/curved/test_easy2.json'  # 输入文件名 # 输入文件名
    output_file = '/home/hsf/FGPrompt/data/datasets/imagenav/mp3d/v1/curved/test_easy.json.gz'  # 输出文件名（.gz格式）

    try:
        # 读取输入文件
        input_data = read_json_or_gzip(input_file)

        # 转换数据格式
        standardized_data = convert_json_format(input_data)

        # 将转换后的数据保存为gzip压缩文件
        save_as_gzip(standardized_data, output_file)

        print(f"转换完成！压缩的标准化数据已保存到 {output_file}")

        # 验证输出文件
        try:
            verification_data = read_json_or_gzip(output_file)
            print("验证成功：输出文件可以正确读取")
        except Exception as e:
            print(f"警告：输出文件验证失败：{str(e)}")

    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file}")
    except json.JSONDecodeError:
        print("错误：输入文件不是有效的JSON格式")
    except Exception as e:
        print(f"发生错误：{str(e)}")


if __name__ == "__main__":
    main()

