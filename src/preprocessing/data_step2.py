import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib
import os
from datetime import datetime, timedelta
#D:\脉氧测量数据\20251103\132844_红光红外信号_processed_full.csv
# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 用户输入功能
def get_user_input():
    """获取用户输入的参数"""
    print("=" * 50)
    print("谷值检测与切片分析工具")
    print("=" * 50)

    # 文件路径输入
    file_path = input("请输入CSV文件路径（默认: F:\\BUAA\\Work\\202509\\0825data\\085240.csv）: ").strip()
    # 移除可能的引号
    file_path = file_path.strip('"\'')
    if not file_path:
        file_path = r"F:\BUAA\Work\202509\0825data\085240.csv"

    # 输出路径输入
    output_path = input("请输入输出Excel文件路径（默认: 与输入文件同目录）: ").strip()
    if not output_path:
        output_dir = os.path.dirname(file_path)
        output_path = os.path.join(output_dir, "slice_analysis_results.xlsx")
    else:
        if not output_path.lower().endswith('.xlsx'):
            output_path += '.xlsx'

    # 初始参考时间输入（年月日时分秒格式）
    print("\n请输入初始参考时间（格式: 2025-09-08 08:52:40）")
    initial_time_str = input("请输入初始参考时间: ").strip()

    try:
        initial_datetime = datetime.strptime(initial_time_str, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        print("时间格式错误，使用默认时间: 2025-09-08 08:52:40")
        initial_datetime = datetime(2025, 8, 25, 8, 52, 40)

    # 采样率输入
    sample_rate_input = input("请输入采样率（Hz，默认: 125）: ").strip()
    try:
        sample_rate = int(sample_rate_input) if sample_rate_input else 125
    except ValueError:
        print("输入无效，使用默认值125")
        sample_rate = 125

    # 切片持续时间输入
    slice_duration_input = input("请输入切片持续时间（秒，默认: 1.0）: ").strip()
    try:
        slice_duration = float(slice_duration_input) if slice_duration_input else 1.0
    except ValueError:
        print("输入无效，使用默认值1.0")
        slice_duration = 1.0

    # 新增用户输入参数
    print("\n请输入以下参数:")
    time_label = input("第一列题头内容（默认: time）: ").strip()
    if not time_label:
        time_label = "time"

    weight_input = input("体重(kg): ").strip()
    try:
        weight = float(weight_input) if weight_input else 70.0
    except ValueError:
        print("输入无效，使用默认值70.0")
        weight = 70.0

    height_input = input("身高(cm): ").strip()
    try:
        height = float(height_input) if height_input else 170.0
    except ValueError:
        print("输入无效，使用默认值170.0")
        height = 170.0

    gender_input = input("性别(M/F): ").strip().upper()
    if gender_input not in ['M', 'F']:
        print("输入无效，使用默认值M")
        gender = 'M'
    else:
        gender = gender_input

    age_input = input("年龄: ").strip()
    try:
        age = int(age_input) if age_input else 30
    except ValueError:
        print("输入无效，使用默认值30")
        age = 30

    id_input = input("ID: ").strip()
    if not id_input:
        id_input = "001"

    # 是否输出谷值识别图片
    output_image_input = input("是否输出谷值识别图片？(y/n，默认: y): ").strip().lower()
    output_image = output_image_input in ['y', 'yes', '是', '']  # 默认为是

    # 图片输出路径
    image_output_path = None
    if output_image:
        image_output_input = input("请输入图片输出路径（默认: 与Excel文件同目录）: ").strip()
        if not image_output_input:
            output_dir = os.path.dirname(output_path)
            image_output_path = os.path.join(output_dir, "valley_detection_plot.png")
        else:
            image_output_path = image_output_input
            if not image_output_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                image_output_path += '.png'

    return (file_path, output_path, initial_datetime, sample_rate, slice_duration,
            output_image, image_output_path, time_label, weight, height, gender, age, id_input)


# 计算BSA (体表面积)
def calculate_bsa(weight, height):
    """使用Du Bois公式计算体表面积"""
    # BSA (m²) = 0.007184 × Weight^0.425 × Height^0.725
    return 0.007184 * (weight ** 0.425) * (height ** 0.725)


# 计算BME (体重指数)
def calculate_bme(weight, height):
    """计算体重指数 (kg/m²)"""
    # BMI = weight (kg) / (height (m))^2
    height_m = height / 100  # 将身高从cm转换为m
    return weight / (height_m ** 2)


# 谷值检测函数
def valley_detection(data):
    """谷值检测"""
    valley_indices, valley_properties = find_peaks(-data, prominence=0.3, distance=20)
    return valley_indices, valley_properties


# 生成谷值识别图片
def generate_valley_detection_plot(time, data, valley_indices, output_path, initial_datetime):
    """生成谷值识别图片"""
    try:
        plt.figure(figsize=(15, 10))

        # 绘制原始数据
        plt.plot(time, data, 'b-', linewidth=1, alpha=0.7, label='脉搏波信号')

        # 标记谷值点
        valley_times = time[valley_indices]
        valley_values = data[valley_indices]
        plt.plot(valley_times, valley_values, 'ro', markersize=6, label=f'检测到的谷值 ({len(valley_indices)}个)')

        # 添加谷值标注
        for i, (t, v) in enumerate(zip(valley_times, valley_values)):
            plt.annotate(f'{i + 1}', (t, v), xytext=(5, 5), textcoords='offset points',
                         fontsize=8, color='red', weight='bold')

        # 设置图表属性
        plt.title('谷值检测结果', fontsize=16, fontweight='bold')
        plt.xlabel('时间 (秒)', fontsize=12)
        plt.ylabel('信号强度', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)

        # 添加统计信息
        stats_text = f'总谷值数: {len(valley_indices)}\n初始时间: {initial_datetime.strftime("%Y-%m-%d %H:%M:%S")}'
        plt.figtext(0.02, 0.02, stats_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

        # 保存图片
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"谷值识别图片已保存到: {output_path}")

    except Exception as e:
        print(f"生成图片时出错: {e}")
        import traceback
        traceback.print_exc()


# 从谷值开始生成切片数据，强制生成225个采集点
def generate_slices_from_valleys(time, data, valley_indices, initial_datetime, sample_rate=225, slice_duration=1.0):
    """从检测到的谷值开始生成切片数据，强制生成225个采集点"""
    slices_data = []
    points_per_slice = int(sample_rate * slice_duration)  # 每个切片需要的点数

    if len(valley_indices) == 0:
        print("未检测到谷值，无法生成切片")
        return slices_data

    # 对每个谷值生成切片
    for i, valley_idx in enumerate(valley_indices):
        # 计算需要的数据点范围
        start_idx = valley_idx
        end_idx = valley_idx + points_per_slice

        # 检查是否超出数据范围
        if end_idx >= len(data):
            print(f"警告: 切片 {i + 1} 需要 {points_per_slice} 个点，但只有 {len(data) - start_idx} 个点可用")
            end_idx = len(data) - 1

        # 获取切片数据
        slice_indices = np.arange(start_idx, end_idx + 1)
        slice_time = time[slice_indices]
        slice_data = data[slice_indices]

        # 计算实际持续时间
        actual_duration = slice_time[-1] - slice_time[0] if len(slice_time) > 1 else 0

        if len(slice_time) > 0:
            datetime_values = [initial_datetime + timedelta(seconds=float(t)) for t in slice_time]

            slices_data.append({
                'slice_index': i + 1,
                'valley_index': valley_idx,
                'start_index': start_idx,
                'end_index': end_idx,
                'start_time': slice_time[0],
                'start_datetime': datetime_values[0],
                'end_time': slice_time[-1],
                'end_datetime': datetime_values[-1],
                'data_points': len(slice_data),
                'datetime_values': datetime_values,
                'time_values': slice_time,
                'data_values': slice_data,
                'valley_value': data[valley_idx],  # 谷值点的数值
                'planned_duration': slice_duration,
                'actual_duration': actual_duration,
                'expected_points': points_per_slice,
                'actual_points': len(slice_data)
            })

    return slices_data


# 计算心率
def calculate_heart_rate(current_slice, next_slice):
    """计算心率（基于两个连续谷值之间的时间差）"""
    if next_slice is None:
        return None

    # 计算两个谷值之间的时间差（秒）
    time_diff = next_slice['start_time'] - current_slice['start_time']

    # 避免除以零
    if time_diff <= 0:
        return None

    # 计算心率（次/分钟）
    heart_rate = 60.0 / time_diff
    return heart_rate


# 保存结果到Excel
def save_results_to_excel(slices_data, output_path, initial_datetime, original_df, valley_indices, file_path,
                          time_label, weight, height, gender, age, id_input):
    """将切片数据保存到Excel文件，包含所有原始采集点"""
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 计算BSA和BME
            bsa = calculate_bsa(weight, height)
            bme = calculate_bme(weight, height)

            # 1. 切片汇总表（横置格式）
            summary_data = []

            # 获取文件名（不含路径）
            file_name = os.path.basename(file_path)

            for i, slice_info in enumerate(slices_data):
                # 计算心率（使用下一个切片的数据）
                next_slice = slices_data[i + 1] if i + 1 < len(slices_data) else None
                heart_rate = calculate_heart_rate(slice_info, next_slice)

                # 创建横置数据行
                row_data = {
                    time_label: file_name,  # 所有行都显示文件名
                    '谷值点索引': slice_info['valley_index'],
                    '起始时间': slice_info['start_datetime'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                    '计算心率': f"{heart_rate:.2f}" if heart_rate is not None else "",
                    'HR': "",  # 第五列题头改为HR
                    'CO': "",  # 第六列题头改为CO
                    'SV': "",  # 第七列题头改为SV
                    'weight': weight,  # 第八列题头改为weight
                    'height': height,  # 第九列题头改为height
                    'gender': gender,  # 第十列题头改为gender
                    'age': age,  # 第十一列题头改为age
                    'BSA': f"{bsa:.4f}",  # 第十二列题头是BSA
                    'BME': f"{bme:.2f}",  # 第十三列题头改为BME
                    'ID': id_input,  # 第十四列题头是ID
                    '信号点数': slice_info['data_points']  # 第十五列题头是信号点数
                }

                # 添加数据值阵列（最多225个点）
                for j, value in enumerate(slice_info['data_values']):
                    if j < 225:  # 限制最多225个数据点
                        row_data[f'signal_{j + 1}'] = value

                summary_data.append(row_data)

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='切片汇总', index=False)

            # 2. 所有切片详细数据表
            all_detailed_data = []
            for slice_info in slices_data:
                for i, (dt, t, d) in enumerate(zip(slice_info['datetime_values'],
                                                   slice_info['time_values'],
                                                   slice_info['data_values'])):
                    all_detailed_data.append({
                        '切片序号': slice_info['slice_index'],
                        '采集点序号': i + 1,
                        '日期时间': dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                        '相对时间(秒)': f"{t:.6f}",
                        '数据数值': d,
                    })

            all_detailed_df = pd.DataFrame(all_detailed_data)
            all_detailed_df.to_excel(writer, sheet_name='所有切片详细数据', index=False)

            # 3. 每个切片的独立工作表
            for slice_info in slices_data:
                slice_detailed_data = []
                for i, (dt, t, d) in enumerate(zip(slice_info['datetime_values'],
                                                   slice_info['time_values'],
                                                   slice_info['data_values'])):
                    slice_detailed_data.append({
                        '采集点序号': i + 1,
                        '日期时间': dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                        '相对时间(秒)': f"{t:.6f}",
                        '数据数值': d,
                        '从起始时间经过(秒)': f"{t - slice_info['start_time']:.6f}",
                        '是否为谷值点': '是' if i == 0 else '否'
                    })

                slice_df = pd.DataFrame(slice_detailed_data)
                sheet_name = f'切片{slice_info["slice_index"]}'
                # Excel工作表名称长度限制为31个字符
                if len(sheet_name) > 31:
                    sheet_name = sheet_name[:31]
                slice_df.to_excel(writer, sheet_name=sheet_name, index=False)

            # 4. 原始数据表（包含切片标记和谷值标记）
            original_with_slice = original_df.copy()
            original_with_slice['切片序号'] = 0
            original_with_slice['采集点在该切片内序号'] = 0
            original_with_slice['是否为谷值点'] = '否'

            # 标记每个数据点属于哪个切片
            for slice_info in slices_data:
                if len(slice_info['time_values']) > 0:
                    # 找到对应索引范围的数据点
                    start_idx = slice_info['start_index']
                    end_idx = slice_info['end_index']
                    indices = np.arange(start_idx, end_idx + 1)

                    if len(indices) > 0:
                        original_with_slice.iloc[indices, original_with_slice.columns.get_loc('切片序号')] = slice_info[
                            'slice_index']
                        original_with_slice.iloc[
                            indices, original_with_slice.columns.get_loc('采集点在该切片内序号')] = range(1,
                                                                                                          len(indices) + 1)

            # 标记谷值点
            if len(valley_indices) > 0:
                original_with_slice.iloc[valley_indices, original_with_slice.columns.get_loc('是否为谷值点')] = '是'

            original_with_slice.to_excel(writer, sheet_name='原始数据带切片标记', index=False)

            # 5. 谷值点信息表
            valley_data = []
            for i, idx in enumerate(valley_indices):
                valley_time = original_df.iloc[idx]['时间(秒)']
                valley_value = original_df.iloc[idx]['处理后的Reddata']
                valley_datetime = initial_datetime + timedelta(seconds=float(valley_time))

                valley_data.append({
                    '谷值序号': i + 1,
                    '时间(秒)': valley_time,
                    '日期时间': valley_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                    '数据数值': valley_value,
                    '数据点索引': idx,
                    '对应切片序号': i + 1 if i < len(slices_data) else '无对应切片'
                })

            valley_df = pd.DataFrame(valley_data)
            valley_df.to_excel(writer, sheet_name='谷值点信息', index=False)

            # 6. 处理信息表
            info_data = {
                '参数': ['初始参考时间', '处理时间', '总切片数', '总数据点数', '检测到谷值数', '输出文件',
                         '体重(kg)', '身高(cm)', '性别', '年龄', 'ID', 'BSA', 'BME'],
                '值': [
                    initial_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    len(slices_data),
                    sum(slice_info['data_points'] for slice_info in slices_data),
                    len(valley_indices),
                    output_path,
                    weight,
                    height,
                    gender,
                    age,
                    id_input,
                    f"{bsa:.4f}",
                    f"{bme:.2f}"
                ]
            }
            info_df = pd.DataFrame(info_data)
            info_df.to_excel(writer, sheet_name='处理信息', index=False)

        print(f"结果已保存到: {output_path}")
        print(f"文件包含以下工作表:")
        print(f"1. 切片汇总 - 横置格式的切片信息")
        print(f"2. 所有切片详细数据 - 所有采集点的完整数据")
        print(f"3. 切片1, 切片2, ... - 每个切片的独立详细数据")
        print(f"4. 原始数据带切片标记 - 原始数据加上切片序号和谷值标记")
        print(f"5. 谷值点信息 - 所有检测到的谷值点详细信息")
        print(f"6. 处理信息 - 处理参数和统计信息")

    except Exception as e:
        print(f"Excel保存失败: {e}")
        import traceback
        traceback.print_exc()


# 主程序
def main():
    # 获取用户输入
    (file_path, output_path, initial_datetime, sample_rate, slice_duration,
     output_image, image_output_path, time_label, weight, height, gender, age, id_input) = get_user_input()

    try:
        # 读取数据
        print(f"正在读取文件: {file_path}")
        df = pd.read_csv(file_path)
        time = df['时间(秒)'].values
        data = df['处理后的Reddata'].values

        print(f"初始参考时间: {initial_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"数据总点数: {len(data)}")
        print(f"总时间范围: {time[0]:.3f} - {time[-1]:.3f} 秒")
        print(f"采样率: {sample_rate} Hz")
        print(f"切片持续时间: {slice_duration} 秒")
        print(f"每个切片期望点数: {int(sample_rate * slice_duration)}")
        print(f"用户参数 - 体重: {weight}kg, 身高: {height}cm, 性别: {gender}, 年龄: {age}, ID: {id_input}")

        # 检测谷值
        print("正在检测谷值...")
        valley_indices, valley_properties = valley_detection(data)

        print(f"检测到 {len(valley_indices)} 个谷值")

        # 生成谷值识别图片
        if output_image and len(valley_indices) > 0:
            print("正在生成谷值识别图片...")
            generate_valley_detection_plot(time, data, valley_indices, image_output_path, initial_datetime)

        # 从谷值开始生成切片数据
        print("正在从谷值开始生成切片数据...")
        slices_data = generate_slices_from_valleys(time, data, valley_indices, initial_datetime, sample_rate,
                                                   slice_duration)

        print(f"\n生成 {len(slices_data)} 个切片")
        for slice_info in slices_data:
            print(f"切片 {slice_info['slice_index']}: "
                  f"起始 {slice_info['start_datetime'].strftime('%H:%M:%S.%f')[:-3]}, "
                  f"数据点 {slice_info['data_points']}/{slice_info['expected_points']}, "
                  f"持续时间 {slice_info['actual_duration']:.3f}秒")

        # 保存结果
        save_results_to_excel(slices_data, output_path, initial_datetime, df, valley_indices, file_path,
                              time_label, weight, height, gender, age, id_input)

        print("\n处理完成！")

    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        print("请检查文件路径和数据格式是否正确")


# 运行主程序
if __name__ == "__main__":
    main()