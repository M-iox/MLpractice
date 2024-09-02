import subprocess
import pandas as pd


def run_task(script_name):
    try:
        subprocess.run(['python', script_name], check=True)
        print(f"{script_name} 运行成功")
    except subprocess.CalledProcessError as e:
        print(f"运行 {script_name} 时出错: {e}")


def merge_csv(output_file, *input_files):
    df_list = []
    for file in input_files:
        df = pd.read_csv(file)
        df_list.append(df)

    # 合并所有数据框
    result_all = pd.concat(df_list, ignore_index=True)
    result_all.to_csv(output_file, index=False)
    print(f"结果已合并并保存至 {output_file}")


if __name__ == "__main__":
    # 运行 task1.py, task2.py, task3.py
    run_task('task1.py')
    run_task('task2.py')
    run_task('task3.py')

    # 合并生成的CSV文件
    merge_csv('result_all.csv', 'result1.csv', 'result2.csv', 'result3.csv')
