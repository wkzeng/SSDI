# statistics_plotting.py
"""
统计和绘图模块 - 从generate_9_methods_and_analyse提取的独立功能
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from datetime import datetime

# 设置matplotlib中文字体
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

# ============================ 统计功能 ============================

def load_data_from_dir(main_out_dir):
    """
    从目录加载数据
    
    参数:
    main_out_dir: 主输出目录
    
    返回:
    combined_df, detailed_stats_df, mechanism_stats_df
    """
    combined_path = os.path.join(main_out_dir, "combined_results.csv")
    detailed_path = os.path.join(main_out_dir, "detailed_statistics.csv")
    mechanism_path = os.path.join(main_out_dir, "mechanism_statistics.csv")
    
    combined_df = pd.read_csv(combined_path) if os.path.exists(combined_path) else None
    detailed_stats_df = pd.read_csv(detailed_path) if os.path.exists(detailed_path) else None
    mechanism_stats_df = pd.read_csv(mechanism_path) if os.path.exists(mechanism_path) else None
    
    return combined_df, detailed_stats_df, mechanism_stats_df

def generate_statistics(data_source, stats_to_generate=None, output_dir=None):
    """
    生成统计信息
    
    参数:
    data_source: 可以是元组 (combined_df, detailed_stats_df, mechanism_stats_df, main_out_dir)
                 或者字符串 main_out_dir
    stats_to_generate: 要生成的统计列表，默认为所有统计
    output_dir: 输出目录，默认为main_out_dir下的stats文件夹
    
    返回:
    统计摘要字典
    """
    # 解析数据源
    if isinstance(data_source, tuple):
        combined_df, detailed_stats_df, mechanism_stats_df, main_out_dir = data_source
    else:
        main_out_dir = data_source
        combined_df, detailed_stats_df, mechanism_stats_df = load_data_from_dir(main_out_dir)
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(main_out_dir, "stats")
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果没有指定要生成的统计，则生成所有统计
    if stats_to_generate is None:
        stats_to_generate = ["A", "B", "C", "D", "E", "F", "G"]
    
    # 初始化统计摘要
    summary_stats = {}
    
    print("="*80)
    print("开始生成统计信息")
    print("="*80)
    
    # 检查数据是否可用
    if combined_df is None:
        print("错误: 无法加载combined_results.csv")
        return summary_stats
    
    # 统计A: LCD机制成功率排名
    if "A" in stats_to_generate:
        print("\n" + "="*80)
        print("统计A: LCD机制成功率排名")
        print("="*80)
        
        lcd_success = {}
        lcd_types = ["client", "class", "joint"]
        
        for lcd_type in lcd_types:
            lcd_data = combined_df[combined_df['lcd_type'] == lcd_type]
            success_rate = lcd_data['success'].mean() * 100 if len(lcd_data) > 0 else 0
            lcd_success[lcd_type] = success_rate
        
        lcd_ranking = sorted(lcd_success.items(), key=lambda x: x[1], reverse=True)
        
        print("LCD Mechanism Success Rate Ranking:")
        for i, (lcd_type, rate) in enumerate(lcd_ranking, 1):
            print(f"{i}. {lcd_type}-LCD: {rate:.1f}%")
        
        best_lcd = lcd_ranking[0][0]
        print(f"\n✓ Best LCD Mechanism: {best_lcd}-LCD ({lcd_ranking[0][1]:.1f}%)")
        
        summary_stats["lcd_ranking"] = lcd_ranking
    
    # 统计B: LDS机制成功率排名
    if "B" in stats_to_generate:
        print("\n" + "="*80)
        print("统计B: LDS机制成功率排名")
        print("="*80)
        
        lds_success = {}
        lds_types = ["client", "special", "lowrank"]
        
        for lds_type in lds_types:
            lds_data = combined_df[combined_df['lds_type'] == lds_type]
            success_rate = lds_data['success'].mean() * 100 if len(lds_data) > 0 else 0
            lds_success[lds_type] = success_rate
        
        lds_ranking = sorted(lds_success.items(), key=lambda x: x[1], reverse=True)
        
        print("LDS Mechanism Success Rate Ranking:")
        for i, (lds_type, rate) in enumerate(lds_ranking, 1):
            print(f"{i}. {lds_type}-LDS: {rate:.1f}%")
        
        best_lds = lds_ranking[0][0]
        print(f"\n✓ Best LDS Mechanism: {best_lds}-LDS ({lds_ranking[0][1]:.1f}%)")
        
        summary_stats["lds_ranking"] = lds_ranking
    
    # 统计C: 9种组合成功率排名
    if "C" in stats_to_generate:
        print("\n" + "="*80)
        print("统计C: 9种机制组合成功率排名")
        print("="*80)
        
        combination_success = {}
        lcd_types = ["client", "class", "joint"]
        lds_types = ["client", "special", "lowrank"]
        
        for lcd_type in lcd_types:
            for lds_type in lds_types:
                combo_data = combined_df[
                    (combined_df['lcd_type'] == lcd_type) &
                    (combined_df['lds_type'] == lds_type)
                ]
                success_rate = combo_data['success'].mean() * 100 if len(combo_data) > 0 else 0
                combination_success[f"{lcd_type}_{lds_type}"] = success_rate
        
        combo_ranking = sorted(combination_success.items(), key=lambda x: x[1], reverse=True)
        
        print("9 Mechanism Combinations Success Rate Ranking:")
        for i, (combo, rate) in enumerate(combo_ranking, 1):
            lcd, lds = combo.split("_")
            print(f"{i}. {lcd}-LCD + {lds}-LDS: {rate:.1f}%")
        
        best_combo = combo_ranking[0][0]
        lcd_best, lds_best = best_combo.split("_")
        print(f"\n✓ Best Combination: {lcd_best}-LCD + {lds_best}-LDS ({combo_ranking[0][1]:.1f}%)")
        
        summary_stats["combo_ranking"] = combo_ranking
    
    # 统计D: 总体统计（增加平均生成时间）
    if "D" in stats_to_generate:
        print("\n" + "="*80)
        print("统计D: 总体统计")
        print("="*80)
        
        total_trials = len(combined_df)
        total_success = combined_df['success'].sum()
        overall_success_rate = total_success / total_trials * 100 if total_trials > 0 else 0
        
        if total_success > 0:
            weighted_avg_iter = combined_df[combined_df['success']]['iter_used'].mean()
            avg_time_elapsed = combined_df[combined_df['success']]['time_elapsed'].mean()
        else:
            weighted_avg_iter = 0
            avg_time_elapsed = 0
        
        print(f"总实验次数: {total_trials}")
        print(f"总成功次数: {total_success}")
        print(f"总体成功率: {overall_success_rate:.1f}%")
        print(f"成功样本的平均迭代次数: {weighted_avg_iter:.1f}")
        print(f"成功样本的平均生成时间: {avg_time_elapsed:.2f} 秒")
        
        summary_stats["overall"] = {
            "total_trials": total_trials,
            "total_success": total_success,
            "overall_success_rate": overall_success_rate,
            "avg_iterations": weighted_avg_iter,
            "avg_time_elapsed": avg_time_elapsed
        }
    
    # 统计E: 全局样本统计
    if "E" in stats_to_generate:
        print("\n" + "="*80)
        print("统计E: 全局样本统计")
        print("="*80)
        
        print("全局统计（所有样本）:")
        print(f"  SSDI范围: {combined_df['SSDI'].min():.3f} - {combined_df['SSDI'].max():.3f}")
        print(f"  LCD范围: {combined_df['LCD'].min():.3f} - {combined_df['LCD'].max():.3f}")
        print(f"  LDS范围: {combined_df['LDS'].min():.3f} - {combined_df['LDS'].max():.3f}")
        print(f"  Missing Rate范围: {combined_df['missing_rate'].min():.3f} - {combined_df['missing_rate'].max():.3f}")
        print(f"  生成时间范围: {combined_df['time_elapsed'].min():.2f} - {combined_df['time_elapsed'].max():.2f} 秒")
        
        summary_stats["global_stats"] = {
            "ssdi_range": (combined_df['SSDI'].min(), combined_df['SSDI'].max()),
            "lcd_range": (combined_df['LCD'].min(), combined_df['LCD'].max()),
            "lds_range": (combined_df['LDS'].min(), combined_df['LDS'].max()),
            "missing_rate_range": (combined_df['missing_rate'].min(), combined_df['missing_rate'].max()),
            "time_range": (combined_df['time_elapsed'].min(), combined_df['time_elapsed'].max())
        }
    
    # 统计F: 成功样本统计
    if "F" in stats_to_generate:
        print("\n" + "="*80)
        print("统计F: 成功样本统计")
        print("="*80)
        
        success_df = combined_df[combined_df['success']]
        
        if len(success_df) > 0:
            print("成功样本统计:")
            print(f"  SSDI范围: {success_df['SSDI'].min():.3f} - {success_df['SSDI'].max():.3f}")
            print(f"  LCD范围: {success_df['LCD'].min():.3f} - {success_df['LCD'].max():.3f}")
            print(f"  LDS范围: {success_df['LDS'].min():.3f} - {success_df['LDS'].max():.3f}")
            print(f"  Missing Rate范围: {success_df['missing_rate'].min():.3f} - {success_df['missing_rate'].max():.3f}")
            print(f"  生成时间范围: {success_df['time_elapsed'].min():.2f} - {success_df['time_elapsed'].max():.2f} 秒")
            
            summary_stats["success_stats"] = {
                "ssdi_range": (success_df['SSDI'].min(), success_df['SSDI'].max()),
                "lcd_range": (success_df['LCD'].min(), success_df['LCD'].max()),
                "lds_range": (success_df['LDS'].min(), success_df['LDS'].max()),
                "missing_rate_range": (success_df['missing_rate'].min(), success_df['missing_rate'].max()),
                "time_range": (success_df['time_elapsed'].min(), success_df['time_elapsed'].max())
            }
        else:
            print("没有成功样本")
            summary_stats["success_stats"] = None
    

    # ============================ 统计G: 详细统计表（拆分成功表与全局表）============================
    if "G" in stats_to_generate and detailed_stats_df is not None:
        print("\n" + "="*80)
        print("统计G1: 成功样本详细统计（按SSDI值）")
        print("="*80)
        
        # ---- 表G1: 成功样本统计 ----
        header_success = (
            "SSDI | 序号 | LCD类型 | LDS类型 | 成功/总数 | 成功率(%) | "
            "迭代次数(成功) | SSDI范围(成功) | LCD范围(成功) | LDS范围(成功) | "
            "实际的alpha(成功) | 实际的beta(成功) | Missing率(成功) | 生成时间(成功)"
        )
        print(header_success)
        print("-" * 240)  # 延长分隔线
        
        for _, row in detailed_stats_df.iterrows():
            # 使用 .get() 避免列不存在时报错
            iter_success = row.get('Iterations_Range_Success', 'N/A')
            ssdi_success = row.get('SSDI_Range_Success', 'N/A')
            lcd_success = row.get('LCD_Range_Success', 'N/A')
            lds_success = row.get('LDS_Range_Success', 'N/A')
            alpha_success = row.get('Alpha_Range_Success', 'N/A')
            beta_success = row.get('Beta_Range_Success', 'N/A')
            missing_success = row.get('Missing_Rate_Range_Success', 'N/A')
            time_success = row.get('Time_Range_Success', 'N/A')
            
            row_str_success = (
                f"{row['SSDI']:6} | "
                f"{row['Index']:3} | "
                f"{row['LCD_Type']:8} | "
                f"{row['LDS_Type']:10} | "
                f"{row['Success_Total']:10} | "
                f"{row['Success_Rate(%)']:10} | "
                f"{iter_success:15} | "
                f"{ssdi_success:15} | "
                f"{lcd_success:15} | "
                f"{lds_success:15} | "
                f"{alpha_success:18} | "
                f"{beta_success:18} | "
                f"{missing_success:15} | "
                f"{time_success:15}"
            )
            print(row_str_success)
        print("-" * 240)

        # ===== 保存G1表为CSV =====
        g1_columns = [
            'SSDI', 'Index', 'LCD_Type', 'LDS_Type', 'Success_Total', 'Success_Rate(%)',
            'Iterations_Range_Success', 'SSDI_Range_Success', 'LCD_Range_Success',
            'LDS_Range_Success', 'Alpha_Range_Success', 'Beta_Range_Success',
            'Missing_Rate_Range_Success', 'Time_Range_Success'
        ]
        g1_exist_cols = [col for col in g1_columns if col in detailed_stats_df.columns]
        g1_df = detailed_stats_df[g1_exist_cols].copy()
        g1_csv_path = os.path.join(output_dir, "detailed_success_statistics.csv")
        g1_df.to_csv(g1_csv_path, index=False, encoding='utf-8')
        print(f"已保存成功样本详细统计表: {g1_csv_path}")



        # ---- 表G2: 全局样本统计 ----
        print("\n" + "="*80)
        print("统计G2: 全局样本详细统计（按SSDI值）")
        print("="*80)
        
        header_global = (
            "SSDI | 序号 | LCD类型 | LDS类型 | "
            "SSDI范围(全局) | LCD范围(全局) | LDS范围(全局) | "
            "实际的alpha(全局) | 实际的beta(全局) | Missing率(全局) | 生成时间(全局)"
        )
        print(header_global)
        print("-" * 200)
        
        for _, row in detailed_stats_df.iterrows():
            iter_global = row.get('Iterations_Range_Global', 'N/A')
            ssdi_global = row.get('SSDI_Range_Global', 'N/A')
            lcd_global = row.get('LCD_Range_Global', 'N/A')
            lds_global = row.get('LDS_Range_Global', 'N/A')
            alpha_global = row.get('Alpha_Range_Global', 'N/A')
            beta_global = row.get('Beta_Range_Global', 'N/A')
            missing_global = row.get('Missing_Rate_Range_Global', 'N/A')
            time_global = row.get('Time_Range_Global', 'N/A')
            
            row_str_global = (
                f"{row['SSDI']:6} | "
                f"{row['Index']:3} | "
                f"{row['LCD_Type']:8} | "
                f"{row['LDS_Type']:10} | "
                f"{ssdi_global:15} | "
                f"{lcd_global:15} | "
                f"{lds_global:15} | "
                f"{alpha_global:18} | "
                f"{beta_global:18} | "
                f"{missing_global:15} | "
                f"{time_global:15}"
            )
            print(row_str_global)
        print("-" * 200)


        # ===== 保存G2表为CSV =====
        g2_columns = [
            'SSDI', 'Index', 'LCD_Type', 'LDS_Type',
            'SSDI_Range_Global', 'LCD_Range_Global', 'LDS_Range_Global',
            'Alpha_Range_Global', 'Beta_Range_Global', 'Missing_Rate_Range_Global',
            'Time_Range_Global'
        ]
        g2_exist_cols = [col for col in g2_columns if col in detailed_stats_df.columns]
        g2_df = detailed_stats_df[g2_exist_cols].copy()
        g2_csv_path = os.path.join(output_dir, "detailed_global_statistics.csv")
        g2_df.to_csv(g2_csv_path, index=False, encoding='utf-8')
        print(f"已保存全局样本详细统计表: {g2_csv_path}")

        # ---- 表G3: 最接近目标SSDI的25%样本统计 ----
        print("\n" + "="*80)
        print("统计G3: 最接近目标SSDI的25%样本详细统计")
        print("="*80)

        # 辅助函数：格式化均值±标准差（处理单样本情况）
        def fmt_mean_std(series, decimals):
            mean = series.mean()
            std = series.std()
            if pd.isna(mean):
                return "N/A"
            if pd.isna(std):
                # 只有一个样本时，标准差为 NaN，显示均值 ± 0
                return f"{mean:.{decimals}f}±0.000"
            return f"{mean:.{decimals}f}±{std:.{decimals}f}"

        # 准备收集G3的行
        g3_rows = []

        # 预处理：为combined_df添加四舍五入的目标SSDI列（避免浮点误差）
        combined_df['target_SSDI_round'] = combined_df['target_SSDI'].round(3)

        # 遍历detailed_stats_df中的每个组合
        for _, row in detailed_stats_df.iterrows():
            C_val = row['C']
            K_val = row['K']
            N_val = row['N']
            ssdi_target = round(float(row['SSDI']), 3)
            lcd_type_val = row['LCD_Type']
            lds_type_val = row['LDS_Type']
            index_val = row['Index']

            # 从combined_df中筛选当前组合的所有样本
            mask = (
                (combined_df['C'] == C_val) &
                (combined_df['K'] == K_val) &
                (combined_df['N'] == N_val) &
                (combined_df['target_SSDI_round'] == ssdi_target) &
                (combined_df['lcd_type'] == lcd_type_val) &
                (combined_df['lds_type'] == lds_type_val)
            )
            combo_samples = combined_df[mask].copy()
            total_samples = len(combo_samples)
            if total_samples == 0:
                continue

            # 计算每个样本的实际SSDI与目标的绝对误差，并取前25%
            combo_samples['error'] = abs(combo_samples['SSDI'] - ssdi_target)
            combo_samples = combo_samples.sort_values('error')
            n_top = max(1, int(np.ceil(total_samples * 0.25)))
            top_samples = combo_samples.head(n_top)

            success_in_top = int(top_samples['success'].sum())

            # 调试输出（可删除）
            #print(f"组合 C={C_val} K={K_val} N={N_val} target={ssdi_target} {lcd_type_val}-{lds_type_val}: 总样本={total_samples}, 取前{n_top}个")

            # 计算各项统计量
            iter_range = fmt_mean_std(top_samples['iter_used'], 1)
            ssdi_range = fmt_mean_std(top_samples['SSDI'], 3)
            lcd_range = fmt_mean_std(top_samples['LCD'], 3)
            lds_range = fmt_mean_std(top_samples['LDS'], 3)
            alpha_range = fmt_mean_std(top_samples['actual_alpha'], 3)
            beta_range = fmt_mean_std(top_samples['actual_beta'], 3)
            missing_range = fmt_mean_std(top_samples['missing_rate'], 3)
            time_range = fmt_mean_std(top_samples['time_elapsed'], 2)

            # 成功/总数 和 成功率(%) 列：填入所选样本数/总样本数
            success_total_str = f"{success_in_top}/{n_top}"
            success_rate_str = f"{success_in_top/n_top*100:.1f}%"

            # 构建行字典
            g3_row = {
                'SSDI': f"{ssdi_target:.3f}",
                'Index': index_val,
                'LCD_Type': lcd_type_val,
                'LDS_Type': lds_type_val,
                'Success_Total': success_total_str,
                'Success_Rate(%)': success_rate_str,
                'Iterations_Range_Success': iter_range,
                'SSDI_Range_Success': ssdi_range,
                'LCD_Range_Success': lcd_range,
                'LDS_Range_Success': lds_range,
                'Alpha_Range_Success': alpha_range,
                'Beta_Range_Success': beta_range,
                'Missing_Rate_Range_Success': missing_range,
                'Time_Range_Success': time_range
            }
            g3_rows.append(g3_row)

        # 转换为DataFrame并输出
        if g3_rows:
            g3_df = pd.DataFrame(g3_rows)

            # 打印表头
            header_g3 = (
                "SSDI | 序号 | LCD类型 | LDS类型 | 成功/总数 | 成功率(%) | "
                "迭代次数(成功) | SSDI范围(成功) | LCD范围(成功) | LDS范围(成功) | "
                "实际的alpha(成功) | 实际的beta(成功) | Missing率(成功) | 生成时间(成功)"
            )
            print(header_g3)
            print("-" * 240)

            for _, row in g3_df.iterrows():
                row_str = (
                    f"{row['SSDI']:6} | "
                    f"{row['Index']:3} | "
                    f"{row['LCD_Type']:8} | "
                    f"{row['LDS_Type']:10} | "
                    f"{row['Success_Total']:10} | "
                    f"{row['Success_Rate(%)']:10} | "
                    f"{row['Iterations_Range_Success']:15} | "
                    f"{row['SSDI_Range_Success']:15} | "
                    f"{row['LCD_Range_Success']:15} | "
                    f"{row['LDS_Range_Success']:15} | "
                    f"{row['Alpha_Range_Success']:18} | "
                    f"{row['Beta_Range_Success']:18} | "
                    f"{row['Missing_Rate_Range_Success']:15} | "
                    f"{row['Time_Range_Success']:15}"
                )
                print(row_str)
            print("-" * 240)

            # 保存为CSV
            g3_csv_path = os.path.join(output_dir, "detailed_nearest25_statistics.csv")
            g3_df.to_csv(g3_csv_path, index=False, encoding='utf-8')
            print(f"已保存最近25%样本详细统计表: {g3_csv_path}")
        else:
            print("警告：没有有效组合用于生成G3表")

    # 保存统计摘要到文件
    summary_path = os.path.join(main_out_dir, "statistics_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("统计摘要\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据目录: {main_out_dir}\n\n")
        
        if "lcd_ranking" in summary_stats:
            f.write("LCD机制成功率排名:\n")
            for i, (lcd_type, rate) in enumerate(summary_stats["lcd_ranking"], 1):
                f.write(f"  {i}. {lcd_type}-LCD: {rate:.1f}%\n")
            f.write("\n")
        
        if "lds_ranking" in summary_stats:
            f.write("LDS机制成功率排名:\n")
            for i, (lds_type, rate) in enumerate(summary_stats["lds_ranking"], 1):
                f.write(f"  {i}. {lds_type}-LDS: {rate:.1f}%\n")
            f.write("\n")
        
        if "combo_ranking" in summary_stats:
            f.write("9种机制组合成功率排名:\n")
            for i, (combo, rate) in enumerate(summary_stats["combo_ranking"], 1):
                lcd, lds = combo.split("_")
                f.write(f"  {i}. {lcd}-LCD + {lds}-LDS: {rate:.1f}%\n")
            f.write("\n")
        
        if "overall" in summary_stats:
            f.write("总体统计:\n")
            f.write(f"  总实验次数: {summary_stats['overall']['total_trials']}\n")
            f.write(f"  总成功次数: {summary_stats['overall']['total_success']}\n")
            f.write(f"  总体成功率: {summary_stats['overall']['overall_success_rate']:.1f}%\n")
            f.write(f"  平均迭代次数: {summary_stats['overall']['avg_iterations']:.1f}\n")
            f.write(f"  平均生成时间: {summary_stats['overall']['avg_time_elapsed']:.2f} 秒\n\n")
    
    print(f"\n统计摘要已保存到: {summary_path}")
    
    return summary_stats

# ============================ 绘图功能 ============================

def _generate_plots_legacy(data_source, plots_to_generate=None, output_dir=None):
    """
    生成图表
    
    参数:
    data_source: 可以是元组 (combined_df, detailed_stats_df, mechanism_stats_df, main_out_dir)
                 或者字符串 main_out_dir
    plots_to_generate: 要生成的图表序号列表，默认为所有图表
    output_dir: 输出目录，默认为main_out_dir下的fig文件夹
    """
    # 解析数据源
    if isinstance(data_source, tuple):
        combined_df, detailed_stats_df, mechanism_stats_df, main_out_dir = data_source
    else:
        main_out_dir = data_source
        combined_df, detailed_stats_df, mechanism_stats_df = load_data_from_dir(main_out_dir)
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(main_out_dir, "fig")
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果没有指定要生成的图表，则生成所有图表
    if plots_to_generate is None:
        plots_to_generate = list(range(1, 11))  # 1到10
    
    print("="*80)
    print("开始生成图表")
    print("="*80)
    
    # 检查数据是否可用
    if combined_df is None:
        print("错误: 无法加载combined_results.csv")
        return
    
    # 获取成功数据
    success_df = combined_df[combined_df['success']].copy()
    
    # 定义映射关系（多个图共用）
    lcd_markers = {'client': 'o', 'class': '^', 'joint': 's'}
    lds_colors = {'client': 'red', 'special': 'blue', 'lowrank': 'yellow'}
    lcd_names = {'client': 'Client-LCD', 'class': 'Class-LCD', 'joint': 'Joint-LCD'}
    lds_names = {'client': 'Client-LDS', 'special': 'Special-LDS', 'lowrank': 'Lowrank-LDS'}
    
    # 绘图1: 每个SSDI值的LCD-LDS散点图
    if 1 in plots_to_generate:
        print("\n生成图1: 各SSDI值的LCD-LDS散点图...")
        
        # 获取唯一的SSDI值
        unique_ssdi = sorted(combined_df['target_SSDI'].unique())
        
        for target_ssdi in unique_ssdi:
            ssdi_data = combined_df[combined_df['target_SSDI'] == target_ssdi]
            
            if ssdi_data.empty:
                continue
            
            plt.figure(figsize=(10, 8))
            
            # 分别绘制成功和失败的点
            for _, row in ssdi_data.iterrows():
                lcd_type = row['lcd_type']
                lds_type = row['lds_type']
                lcd_val = row['LCD']
                lds_val = row['LDS']
                success = row['success']
                
                marker = lcd_markers[lcd_type]
                color = lds_colors[lds_type]
                
                if success:
                    face_color = color
                    edge_color = color
                    marker_size = 80
                    alpha = 0.8
                else:
                    face_color = 'none'
                    edge_color = color
                    marker_size = 60
                    alpha = 0.4
                
                plt.scatter(lcd_val, lds_val,
                           marker=marker,
                           s=marker_size,
                           facecolors=face_color,
                           edgecolors=edge_color,
                           alpha=alpha,
                           linewidths=1.5)
            
            # 设置图表属性
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.xlabel('LCD Value', fontsize=12, fontweight='bold')
            plt.ylabel('LDS Value', fontsize=12, fontweight='bold')
            plt.title(f'LCD-LDS Scatter Plot (Target SSDI={target_ssdi:.3f})', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3, linestyle='--')
            
            # 添加对角线
            x = np.linspace(0, 1, 100)
            for ssdi_line in [0.2, 0.4, 0.6, 0.8]:
                y = np.sqrt(np.maximum(0, ssdi_line**2 - x**2))
                plt.plot(x, y, 'k--', alpha=0.3, linewidth=0.8)
            
            # 创建图例
            legend_elements = []
            for lcd_type, marker in lcd_markers.items():
                legend_elements.append(Line2D([0], [0],
                                            marker=marker,
                                            color='w',
                                            label=lcd_names[lcd_type],
                                            markerfacecolor='gray',
                                            markersize=10))
            
            for lds_type, color in lds_colors.items():
                legend_elements.append(Patch(facecolor=color,
                                           label=lds_names[lds_type],
                                           alpha=0.8))
            
            legend_elements.append(Line2D([0], [0],
                                        marker='o',
                                        color='w',
                                        label='Success',
                                        markerfacecolor='black',
                                        markersize=10))
            legend_elements.append(Line2D([0], [0],
                                        marker='o',
                                        color='w',
                                        label='Failure',
                                        markerfacecolor='none',
                                        markeredgecolor='black',
                                        markersize=10))
            
            plt.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
            
            # 保存图片
            fig_path = os.path.join(output_dir, f'scatter_ssdi_{target_ssdi:.3f}.png')
            plt.tight_layout()
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  已保存: {fig_path}")



    # 绘图22: 所有样本的LCD-LDS散点图（成功实心，失败空心）
    if 22 in plots_to_generate:
        print("\n生成图22: 所有样本的LCD-LDS散点图（成功实心，失败空心）...")
        
        plt.figure(figsize=(12, 10))
        
        # 遍历所有样本
        for _, row in combined_df.iterrows():
            lcd_type = row['lcd_type']
            lds_type = row['lds_type']
            lcd_val = row['LCD']
            lds_val = row['LDS']
            success = row['success']
            
            marker = lcd_markers[lcd_type]
            color = lds_colors[lds_type]
            
            if success:
                face_color = color
                edge_color = color
                marker_size = 80
                alpha = 0.8
            else:
                face_color = 'none'
                edge_color = color
                marker_size = 60
                alpha = 0.4
            
            plt.scatter(lcd_val, lds_val,
                       marker=marker,
                       s=marker_size,
                       facecolors=face_color,
                       edgecolors=edge_color,
                       alpha=alpha,
                       linewidths=1.5)
        
        # 设置图表属性
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('LCD Value', fontsize=14, fontweight='bold')
        plt.ylabel('LDS Value', fontsize=14, fontweight='bold')
        plt.title('All Samples: LCD-LDS Scatter Plot (Success Solid, Failure Hollow)', 
                  fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # 添加对角线（SSDI等值线）
        x = np.linspace(0, 1, 100)
        for ssdi_line in [0.2, 0.4, 0.6, 0.8]:
            y = np.sqrt(np.maximum(0, ssdi_line**2 - x**2))
            plt.plot(x, y, 'k--', alpha=0.3, linewidth=0.8)
        
        # 创建图例
        legend_elements = []
        for lcd_type, marker in lcd_markers.items():
            legend_elements.append(Line2D([0], [0],
                                        marker=marker,
                                        color='w',
                                        label=lcd_names[lcd_type],
                                        markerfacecolor='gray',
                                        markersize=12))
        
        for lds_type, color in lds_colors.items():
            legend_elements.append(Patch(facecolor=color,
                                       label=lds_names[lds_type],
                                       alpha=0.8))
        
        legend_elements.append(Line2D([0], [0],
                                    marker='o',
                                    color='w',
                                    label='Success',
                                    markerfacecolor='black',
                                    markersize=12))
        legend_elements.append(Line2D([0], [0],
                                    marker='o',
                                    color='w',
                                    label='Failure',
                                    markerfacecolor='none',
                                    markeredgecolor='black',
                                    markersize=12))
        
        plt.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)
        
        # 保存图片
        fig_path = os.path.join(output_dir, 'scatter_all_samples.png')
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {fig_path}")




    
    # 绘图2: 所有成功样本的LCD-LDS散点图
    if 2 in plots_to_generate and not success_df.empty:
        print("\n生成图2: 所有成功样本的LCD-LDS散点图...")
        
        plt.figure(figsize=(12, 10))
        
        for _, row in success_df.iterrows():
            lcd_type = row['lcd_type']
            lds_type = row['lds_type']
            lcd_val = row['LCD']
            lds_val = row['LDS']
            target_ssdi = row['target_SSDI']
            
            marker = lcd_markers[lcd_type]
            color = lds_colors[lds_type]
            point_size = 50 + target_ssdi * 100
            
            plt.scatter(lcd_val, lds_val,
                       marker=marker,
                       s=point_size,
                       facecolors=color,
                       edgecolors='black',
                       alpha=0.7,
                       linewidths=1.0)
        
        # 设置图表属性
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('LCD Value', fontsize=14, fontweight='bold')
        plt.ylabel('LDS Value', fontsize=14, fontweight='bold')
        plt.title('All Successful Samples: LCD-LDS Scatter Plot', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # 添加对角线
        x = np.linspace(0, 1, 100)
        for ssdi_line in [0.2, 0.4, 0.6, 0.8, 1.0]:
            y = np.sqrt(np.maximum(0, ssdi_line**2 - x**2))
            plt.plot(x, y, 'k--', alpha=0.3, linewidth=1.0)
            if ssdi_line <= 0.8:
                plt.text(ssdi_line*0.95, 0.02, f'SSDI={ssdi_line}',
                        fontsize=9, alpha=0.7, rotation=45)
        
        # 创建图例
        legend_elements = []
        for lcd_type, marker in lcd_markers.items():
            legend_elements.append(Line2D([0], [0],
                                        marker=marker,
                                        color='w',
                                        label=lcd_names[lcd_type],
                                        markerfacecolor='gray',
                                        markersize=12))
        
        for lds_type, color in lds_colors.items():
            legend_elements.append(Patch(facecolor=color,
                                       label=lds_names[lds_type],
                                       alpha=0.8))
        
        plt.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)
        
        # 保存图片
        fig_path = os.path.join(output_dir, 'scatter_all_success.png')
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  已保存: {fig_path}")
    

    if 33 in plots_to_generate:
        print("\n生成图33: 所有样本的Alpha-Beta参数空间图（成功实心，失败空心）...")
        
        # 过滤掉actual_alpha/actual_beta为NaN的行
        plot_df = combined_df.dropna(subset=['actual_alpha', 'actual_beta']).copy()
        
        if not plot_df.empty:
            plt.figure(figsize=(12, 10))
            
            # 准备颜色映射
            cmap = plt.cm.viridis
            norm = plt.Normalize(plot_df['SSDI'].min(), plot_df['SSDI'].max())
            colors = cmap(norm(plot_df['SSDI']))
            
            # 准备点大小（缩放因子可调整）
            sizes = 20 + plot_df['target_SSDI'] * 250
            
                
            # ---- 定义成功/失败掩码 ----
            success_mask = plot_df['success'].astype(bool)
            failure_mask = ~success_mask                

            # 绘制成功样本（实心圆，黑色边框）
            if success_mask.any():
                plt.scatter(
                    plot_df.loc[success_mask, 'actual_alpha'],
                    plot_df.loc[success_mask, 'actual_beta'],
                    c=colors[success_mask],
                    s=sizes[success_mask],
                    marker='o',
                    facecolors='none',  # 实际由 c 参数控制
                    edgecolors='black',
                    linewidths=0.5,
                    alpha=0.8,
                    label='Success'
                )
            
            # 绘制失败样本（空心圆，边框为实际SSDI颜色）
            if failure_mask.any():
                plt.scatter(
                    plot_df.loc[failure_mask, 'actual_alpha'],
                    plot_df.loc[failure_mask, 'actual_beta'],
                    c='none',                        # 无填充
                    s=sizes[failure_mask],
                    marker='o',
                    #facecolors='none',
                    edgecolors=colors[failure_mask],  # 边框颜色 = 实际SSDI颜色
                    linewidths=1.0,
                    alpha=0.6,
                    label='Failure'
                )



            # 设置坐标轴（对数刻度）
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Estimated Pareto Alpha (log scale)', fontsize=14, fontweight='bold')
            plt.ylabel('Estimated Zipf Beta (log scale)', fontsize=14, fontweight='bold')
            plt.title('All Samples: Alpha-Beta Parameter Space\n(Color = Actual SSDI, Size = Target SSDI)',
                      fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.3, linestyle='--', which='both')
            
            # 添加颜色条
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array(plot_df['SSDI'])  
            cbar = plt.colorbar(sm, ax=plt.gca())
            cbar.set_label('Actual SSDI Value', fontsize=12, fontweight='bold')
            cbar.ax.tick_params(labelsize=10)
            
            # 添加图例（成功/失败）
            plt.legend(loc='upper right', fontsize=12, framealpha=0.9)
            
            # 可选：添加目标SSDI大小示例图例
            target_ssdi_values = [0.2, 0.4, 0.6, 0.8]
            size_legend_elements = []
            for ssdi_val in target_ssdi_values:
                size = 20 + ssdi_val * 250
                size_legend_elements.append(
                    Line2D([0], [0], marker='o', color='w',
                          label=f'Target SSDI={ssdi_val:.1f}',
                          markerfacecolor='gray', markersize=np.sqrt(size)/5,
                          markeredgewidth=0.5)
                )
            # 将大小图例放在左下角
            legend2 = plt.legend(handles=size_legend_elements, loc='lower left',
                                title='Point Size (Target SSDI)', fontsize=10,
                                title_fontsize=11, framealpha=0.9)
            plt.gca().add_artist(legend2)  # 保留图例
            
            # 保存图片
            fig_path = os.path.join(output_dir, 'alpha_beta_all_samples.png')
            plt.tight_layout()
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  已保存: {fig_path}")
        else:
            print("  警告: 没有可用的Alpha/Beta数据，跳过绘图33")
    
    # 绘图4-10 原有代码保持不变...
    # ...
    
    print("\n图表生成完成！")



    # 绘图3: Alpha-Beta参数空间图
    if 3 in plots_to_generate and not success_df.empty:
        print("\n生成图3: Alpha-Beta参数空间图...")
        
        plot_df = success_df.dropna(subset=['actual_alpha', 'actual_beta', 'SSDI', 'target_SSDI'])
        
        if not plot_df.empty:
            plt.figure(figsize=(12, 10))
            
            scatter = plt.scatter(plot_df['actual_alpha'],
                                 plot_df['actual_beta'],
                                 c=plot_df['SSDI'],
                                 s=50 + plot_df['target_SSDI'] * 100,
                                 cmap='viridis',
                                 alpha=0.7,
                                 edgecolors='black',
                                 linewidths=0.5)
            
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Estimated Pareto Alpha (log scale)', fontsize=14, fontweight='bold')
            plt.ylabel('Estimated Zipf Beta (log scale)', fontsize=14, fontweight='bold')
            plt.title('Alpha-Beta Parameter Space with SSDI Color Coding', fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.3, linestyle='--', which='both')
            
            cbar = plt.colorbar(scatter)
            cbar.set_label('Actual SSDI Value', fontsize=12, fontweight='bold')
            cbar.ax.tick_params(labelsize=10)
            
            target_ssdi_values = [0.2, 0.5, 0.8]
            legend_elements = []
            for ssdi_val in target_ssdi_values:
                size = 50 + ssdi_val * 100
                legend_elements.append(Line2D([0], [0],
                                            marker='o',
                                            color='w',
                                            label=f'Target SSDI={ssdi_val:.1f}',
                                            markerfacecolor='gray',
                                            markersize=np.sqrt(size)/5,
                                            markeredgewidth=0.5))
            
            plt.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.9)
            
            fig_path = os.path.join(output_dir, 'alpha_beta_parameter_space.png')
            plt.tight_layout()
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  已保存: {fig_path}")
    
    # 绘图4: 机制组合成功率热力图
    if 4 in plots_to_generate:
        print("\n生成图4: 机制组合成功率热力图...")
        
        # 创建机制组合成功率矩阵
        mechanism_matrix = np.zeros((3, 3))
        lcd_labels = ['client', 'class', 'joint']
        lds_labels = ['client', 'special', 'lowrank']
        
        for i, lcd_type in enumerate(lcd_labels):
            for j, lds_type in enumerate(lds_labels):
                combo_data = combined_df[
                    (combined_df['lcd_type'] == lcd_type) &
                    (combined_df['lds_type'] == lds_type)
                ]
                if len(combo_data) > 0:
                    success_rate = combo_data['success'].mean() * 100
                    mechanism_matrix[i, j] = success_rate
        
        plt.figure(figsize=(10, 8))
        im = plt.imshow(mechanism_matrix, cmap='YlOrRd', aspect='auto')
        
        plt.xticks(np.arange(len(lds_labels)), [lds_names[lds] for lds in lds_labels], fontsize=11, rotation=45)
        plt.yticks(np.arange(len(lcd_labels)), [lcd_names[lcd] for lcd in lcd_labels], fontsize=11)
        
        # 添加数值标签
        for i in range(len(lcd_labels)):
            for j in range(len(lds_labels)):
                text = plt.text(j, i, f'{mechanism_matrix[i, j]:.1f}%',
                               ha="center", va="center",
                               color="white" if mechanism_matrix[i, j] > 50 else "black",
                               fontsize=12, fontweight='bold')
        
        plt.xlabel('LDS Mechanism', fontsize=13, fontweight='bold')
        plt.ylabel('LCD Mechanism', fontsize=13, fontweight='bold')
        plt.title('Success Rate Heatmap of LCD×LDS Mechanisms', fontsize=15, fontweight='bold')
        
        cbar = plt.colorbar(im)
        cbar.set_label('Success Rate (%)', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)
        
        fig_path = os.path.join(output_dir, 'mechanism_success_heatmap.png')
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  已保存: {fig_path}")
    
    # 绘图5: 每个SSDI的3×3子图超大图
    if 5 in plots_to_generate and not success_df.empty:
        print("\n生成图5: 各SSDI的3×3机制子图...")
        
        unique_ssdi = sorted(success_df['target_SSDI'].unique())
        
        for ssdi_val in unique_ssdi:
            ssdi_data = combined_df[combined_df['target_SSDI'] == ssdi_val]
            
            if ssdi_data.empty:
                continue
            
            fig, axes = plt.subplots(3, 3, figsize=(18, 16), sharex=True, sharey=True)
            fig.suptitle(f'LCD-LDS Scatter Plots for Target SSDI = {ssdi_val:.3f}',
                        fontsize=20, fontweight='bold', y=0.95)
            
            lcd_order = ['client', 'class', 'joint']
            lds_order = ['client', 'special', 'lowrank']
            
            for i, lcd_type in enumerate(lcd_order):
                for j, lds_type in enumerate(lds_order):
                    ax = axes[i, j]
                    
                    combo_data = ssdi_data[
                        (ssdi_data['lcd_type'] == lcd_type) &
                        (ssdi_data['lds_type'] == lds_type)
                    ]
                    
                    if combo_data.empty:
                        ax.text(0.5, 0.5, 'No Data',
                              ha='center', va='center', fontsize=12)
                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1)
                    else:
                        success_data = combo_data[combo_data['success']]
                        failure_data = combo_data[~combo_data['success']]
                        
                        if not failure_data.empty:
                            ax.scatter(failure_data['LCD'], failure_data['LDS'],
                                      facecolors='none', edgecolors='gray',
                                      s=15, alpha=0.3, linewidths=0.5)
                        
                        if not success_data.empty:
                            color = 'red' if lds_type == 'client' else ('blue' if lds_type == 'special' else 'yellow')
                            ax.scatter(success_data['LCD'], success_data['LDS'],
                                      facecolors=color, edgecolors='black',
                                      s=25, alpha=0.8, linewidths=0.5)
                    
                    ax.set_title(f'{lcd_type}-LCD × {lds_type}-LDS',
                                fontsize=14, fontweight='bold', pad=10)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    
                    if i == 2:
                        ax.set_xlabel('LCD Value', fontsize=12, fontweight='bold')
                    if j == 0:
                        ax.set_ylabel('LDS Value', fontsize=12, fontweight='bold')
                    
                    ax.grid(True, alpha=0.2, linestyle='--')
                    
                    x_line = np.linspace(0, 1, 100)
                    y_line = np.sqrt(np.maximum(0, ssdi_val**2 - x_line**2))
                    ax.plot(x_line, y_line, 'k--', alpha=0.3, linewidth=1)
            
            plt.subplots_adjust(wspace=0.15, hspace=0.25, top=0.92, bottom=0.05)
            
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', label='Success',
                          markerfacecolor='red', markersize=8, markeredgecolor='black'),
                plt.Line2D([0], [0], marker='o', color='w', label='Failure',
                          markerfacecolor='none', markersize=8, markeredgecolor='gray')
            ]
            
            fig.legend(handles=legend_elements, loc='lower center',
                      ncol=2, fontsize=12, framealpha=0.9)
            
            fig_path = os.path.join(output_dir, f'subplot_matrix_ssdi_{ssdi_val:.3f}.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"  已保存: {fig_path}")
    
    # 绘图6: 所有SSDI的3×3子图超大图
    if 6 in plots_to_generate:
        print("\n生成图6: 所有SSDI的3×3机制子图...")
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 16), sharex=True, sharey=True)
        fig.suptitle('LCD-LDS Scatter Plots for All SSDI Values',
                    fontsize=20, fontweight='bold', y=0.95)
        
        lcd_order = ['client', 'class', 'joint']
        lds_order = ['client', 'special', 'lowrank']
        
        for i, lcd_type in enumerate(lcd_order):
            for j, lds_type in enumerate(lds_order):
                ax = axes[i, j]
                
                combo_data = combined_df[
                    (combined_df['lcd_type'] == lcd_type) &
                    (combined_df['lds_type'] == lds_type)
                ]
                
                if combo_data.empty:
                    ax.text(0.5, 0.5, 'No Data',
                          ha='center', va='center', fontsize=12)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                else:
                    success_data = combo_data[combo_data['success']]
                    failure_data = combo_data[~combo_data['success']]
           
         
                    if not failure_data.empty:
                        ax.scatter(failure_data['LCD'], failure_data['LDS'],
                                  facecolors='none', edgecolors='gray',
                                  s=18, alpha=0.45, linewidths=0.7, zorder=1)
                    
                    if not success_data.empty:
                        color = 'red' if lds_type == 'client' else ('blue' if lds_type == 'special' else 'yellow')
                        scatter = ax.scatter(success_data['LCD'], success_data['LDS'],
                                            facecolors=color, edgecolors='black',
                                            s=15, alpha=0.7, linewidths=0.3, zorder=2)
                        
                        n_success = len(success_data)
                        n_fail = len(failure_data)
                        ax.text(0.05, 0.95, f's={n_success}\nf={n_fail}',
                              transform=ax.transAxes, fontsize=10,
                              verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))

                
                ax.set_title(f'{lcd_type}-LCD × {lds_type}-LDS',
                            fontsize=14, fontweight='bold', pad=10)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                
                if i == 2:
                    ax.set_xlabel('LCD Value', fontsize=12, fontweight='bold')
                if j == 0:
                    ax.set_ylabel('LDS Value', fontsize=12, fontweight='bold')
                
                ax.grid(True, alpha=0.15, linestyle='--')
                
                x_line = np.linspace(0, 1, 100)
                for ssdi_val in [0.2, 0.4, 0.6, 0.8]:
                    y_line = np.sqrt(np.maximum(0, ssdi_val**2 - x_line**2))
                    ax.plot(x_line, y_line, 'k--', alpha=0.2, linewidth=0.5, zorder=0)
        
        plt.subplots_adjust(wspace=0.15, hspace=0.25, top=0.92, bottom=0.05)
        
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Success (All SSDI)',
                      markerfacecolor='red', markersize=8, markeredgecolor='black'),
            plt.Line2D([0], [0], marker='o', color='w', label='Failure (All SSDI)',
                      markerfacecolor='none', markersize=8, markeredgecolor='gray')
        ]
        
        fig.legend(handles=legend_elements, loc='lower center',
                  ncol=2, fontsize=12, framealpha=0.9)
        
        fig_path = os.path.join(output_dir, 'subplot_matrix_all_ssdi.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  已保存: {fig_path}")
    
    # 绘图7: 机制成功率柱状图
    if 7 in plots_to_generate:
        print("\n生成图7: 机制组合成功率柱状图...")
        
        combo_success_rates = []
        combo_labels = []
        lcd_order = ['client', 'class', 'joint']
        lds_order = ['client', 'special', 'lowrank']
        
        for lcd_type in lcd_order:
            for lds_type in lds_order:
                combo_data = combined_df[
                    (combined_df['lcd_type'] == lcd_type) &
                    (combined_df['lds_type'] == lds_type)
                ]
                
                if len(combo_data) > 0:
                    success_rate = combo_data['success'].mean() * 100
                    combo_success_rates.append(success_rate)
                    combo_labels.append(f'{lcd_type}\n{lds_type}')
        
        plt.figure(figsize=(14, 8))
        bars = plt.bar(range(len(combo_success_rates)), combo_success_rates)
        
        colors = []
        for i, label in enumerate(combo_labels):
            lcd_type = label.split('\n')[0]
            if lcd_type == 'client':
                colors.append('lightcoral')
            elif lcd_type == 'class':
                colors.append('lightblue')
            else:
                colors.append('lightyellow')
        
        for i, bar in enumerate(bars):
            bar.set_color(colors[i])
            bar.set_edgecolor('black')
            bar.set_linewidth(1)
            
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{combo_success_rates[i]:.1f}%', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
        
        plt.xticks(range(len(combo_success_rates)), combo_labels, fontsize=11, rotation=0)
        plt.xlabel('Mechanism Combination (LCD × LDS)', fontsize=14, fontweight='bold')
        plt.ylabel('Success Rate (%)', fontsize=14, fontweight='bold')
        plt.title('Success Rates for Different LCD-LDS Combinations',
                fontsize=16, fontweight='bold')
        plt.ylim(0, max(combo_success_rates) * 1.2 if combo_success_rates else 100)
        plt.grid(True, alpha=0.3, axis='y')
        
        lcd_legend_elements = [
            Patch(facecolor='lightcoral', edgecolor='black', label='Client-LCD'),
            Patch(facecolor='lightblue', edgecolor='black', label='Class-LCD'),
            Patch(facecolor='lightyellow', edgecolor='black', label='Joint-LCD')
        ]
        
        plt.legend(handles=lcd_legend_elements, loc='upper right', fontsize=12)
        
        fig_path = os.path.join(output_dir, 'mechanism_success_bar.png')
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  已保存: {fig_path}")
    
    # 绘图8: 九条折线图
    if 8 in plots_to_generate:
        print("\n生成图8: 九条折线图（颜色=LCD，线宽=LDS）...")
        
        success_rate_by_ssdi = combined_df.groupby(['target_SSDI', 'lcd_type', 'lds_type'])['success'].mean() * 100
        success_rate_by_ssdi = success_rate_by_ssdi.reset_index()
        
        plt.figure(figsize=(14, 10))
        
        lcd_colors = {
            'client': '#E74C3C',
            'class': '#3498DB',
            'joint': '#2ECC71'
        }
        
        lds_linewidths = {'client': 1.5, 'special': 2.5, 'lowrank': 3.5}
        lds_linestyles = {'client': '-', 'special': '--', 'lowrank': '-.'}
        lds_markers = {'client': 'o', 'special': 's', 'lowrank': '^'}
        
        ssdi_values = sorted(success_rate_by_ssdi['target_SSDI'].unique())
        
        for lcd_type in ['client', 'class', 'joint']:
            for lds_type in ['client', 'special', 'lowrank']:
                combo_data = success_rate_by_ssdi[
                    (success_rate_by_ssdi['lcd_type'] == lcd_type) &
                    (success_rate_by_ssdi['lds_type'] == lds_type)
                ]
                
                if not combo_data.empty:
                    combo_data = combo_data.sort_values('target_SSDI')
                    rates = combo_data['success'].values
                    rates = np.where(pd.isna(rates), 0, rates)
                    
                    plt.plot(combo_data['target_SSDI'], rates,
                            color=lcd_colors[lcd_type],
                            linewidth=lds_linewidths[lds_type],
                            linestyle=lds_linestyles[lds_type],
                            marker=lds_markers[lds_type],
                            markersize=8,
                            markeredgecolor='black',
                            markeredgewidth=0.5,
                            label=f'{lcd_type}-LCD × {lds_type}-LDS',
                            alpha=0.8)
        
        plt.xlabel('Target SSDI Value', fontsize=14, fontweight='bold')
        plt.ylabel('Success Rate (%)', fontsize=14, fontweight='bold')
        plt.title('Success Rates of Nine Mechanism Combinations\n(Color: LCD Mechanism, Line Width: LDS Mechanism)',
                fontsize=16, fontweight='bold', pad=20)
        
        plt.xlim(min(ssdi_values) - 0.02, max(ssdi_values) + 0.02)
        plt.ylim(-2, 102)
        plt.xticks(ssdi_values, [f'{x:.2f}' for x in ssdi_values], fontsize=11)
        plt.yticks(np.arange(0, 101, 10), fontsize=11)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        lcd_legend_elements = [
            Line2D([0], [0], color=lcd_colors['client'], lw=3, label='Client-LCD'),
            Line2D([0], [0], color=lcd_colors['class'], lw=3, label='Class-LCD'),
            Line2D([0], [0], color=lcd_colors['joint'], lw=3, label='Joint-LCD')
        ]
        
        lds_legend_elements = [
            Line2D([0], [0], color='black', lw=lds_linewidths['client'],
                  linestyle=lds_linestyles['client'], label='Client-LDS (thin)'),
            Line2D([0], [0], color='black', lw=lds_linewidths['special'],
                  linestyle=lds_linestyles['special'], label='Special-LDS (medium)'),
            Line2D([0], [0], color='black', lw=lds_linewidths['lowrank'],
                  linestyle=lds_linestyles['lowrank'], label='Lowrank-LDS (thick)')
        ]
        
        lcd_legend = plt.legend(handles=lcd_legend_elements, loc='upper left',
                              title='LCD Mechanism (Color)', fontsize=11,
                              title_fontsize=12, framealpha=0.9)
        plt.gca().add_artist(lcd_legend)
        
        lds_legend = plt.legend(handles=lds_legend_elements, loc='upper right',
                              title='LDS Mechanism (Line Width/Style)', fontsize=11,
                              title_fontsize=12, framealpha=0.9)
        
        overall_success_rate = combined_df['success'].mean() * 100
        plt.axhline(y=overall_success_rate, color='gray', linestyle=':',
                  linewidth=1.5, alpha=0.7, label=f'Overall Avg: {overall_success_rate:.1f}%')
        
        fig_path = os.path.join(output_dir, 'nine_line_chart_success_rates.png')
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  已保存: {fig_path}")
    
    # 绘图9: 改进版折线图
    if 9 in plots_to_generate:
        print("\n生成图9: 带填充区域的改进版折线图...")
        
        success_rate_by_ssdi = combined_df.groupby(['target_SSDI', 'lcd_type', 'lds_type'])['success'].mean() * 100
        success_rate_by_ssdi = success_rate_by_ssdi.reset_index()
        
        plt.figure(figsize=(16, 10))
        
        lcd_colors = {
            'client': '#E74C3C',
            'class': '#3498DB',
            'joint': '#2ECC71'
        }
        
        lds_linewidths = {'client': 1.5, 'special': 2.5, 'lowrank': 3.5}
        lds_linestyles = {'client': '-', 'special': '--', 'lowrank': '-.'}
        lds_markers = {'client': 'o', 'special': 's', 'lowrank': '^'}
        
        ssdi_values = sorted(success_rate_by_ssdi['target_SSDI'].unique())
        
        for lcd_type in ['client', 'class', 'joint']:
            for lds_type in ['client', 'special', 'lowrank']:
                combo_data = success_rate_by_ssdi[
                    (success_rate_by_ssdi['lcd_type'] == lcd_type) &
                    (success_rate_by_ssdi['lds_type'] == lds_type)
                ]
                
                if not combo_data.empty:
                    combo_data = combo_data.sort_values('target_SSDI')
                    rates = combo_data['success'].values
                    rates = np.where(pd.isna(rates), 0, rates)
                    
                    line, = plt.plot(combo_data['target_SSDI'], rates,
                                    color=lcd_colors[lcd_type],
                                    linewidth=lds_linewidths[lds_type],
                                    linestyle=lds_linestyles[lds_type],
                                    marker=lds_markers[lds_type],
                                    markersize=9,
                                    markeredgecolor='black',
                                    markeredgewidth=0.8,
                                    label=f'{lcd_type}×{lds_type}',
                                    alpha=0.9)
                    
                    plt.fill_between(combo_data['target_SSDI'], 0, rates,
                                    color=lcd_colors[lcd_type],
                                    alpha=0.1)
        
        plt.xlabel('Target SSDI Value', fontsize=14, fontweight='bold')
        plt.ylabel('Success Rate (%)', fontsize=14, fontweight='bold')
        plt.title('Success Rates with Confidence Regions\n(Color: LCD Mechanism, Line Width/Style: LDS Mechanism)',
                fontsize=16, fontweight='bold', pad=20)
        
        plt.xlim(min(ssdi_values) - 0.02, max(ssdi_values) + 0.02)
        plt.ylim(-5, 105)
        plt.xticks(ssdi_values, [f'{x:.2f}' for x in ssdi_values], fontsize=11)
        plt.yticks(np.arange(0, 101, 10), fontsize=11)
        plt.grid(True, alpha=0.25, linestyle='--', which='both')
        
        combined_legend_elements = []
        
        for lcd_type, color in lcd_colors.items():
            combined_legend_elements.append(
                Line2D([0], [0], color=color, lw=3, label=f'{lcd_type}-LCD')
            )
        
        combined_legend_elements.append(Line2D([0], [0], color='none', label=''))
        
        for lds_type in ['client', 'special', 'lowrank']:
            combined_legend_elements.append(
                Line2D([0], [0], color='black',
                      lw=lds_linewidths[lds_type],
                      linestyle=lds_linestyles[lds_type],
                      label=f'{lds_type}-LDS')
            )
        
        plt.legend(handles=combined_legend_elements, loc='upper left',
                  fontsize=11, framealpha=0.95, ncol=2)
        
        fig_path = os.path.join(output_dir, 'nine_line_chart_with_regions.png')
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  已保存: {fig_path}")
    
    # 绘图10: 交互式风格折线图
    if 10 in plots_to_generate:
        print("\n生成图10: 交互式风格折线图（带详细数据点）...")
        
        success_rate_by_ssdi = combined_df.groupby(['target_SSDI', 'lcd_type', 'lds_type'])['success'].mean() * 100
        success_rate_by_ssdi = success_rate_by_ssdi.reset_index()
        
        fig, ax = plt.subplots(figsize=(18, 12))
        
        lcd_colors = {
            'client': '#E74C3C',
            'class': '#3498DB',
            'joint': '#2ECC71'
        }
        
        lds_linewidths = {'client': 1.5, 'special': 2.5, 'lowrank': 3.5}
        lds_linestyles = {'client': '-', 'special': '--', 'lowrank': '-.'}
        lds_markers = {'client': 'o', 'special': 's', 'lowrank': '^'}
        
        ssdi_values = sorted(success_rate_by_ssdi['target_SSDI'].unique())
        
        for lcd_type in ['client', 'class', 'joint']:
            for lds_type in ['client', 'special', 'lowrank']:
                combo_data = success_rate_by_ssdi[
                    (success_rate_by_ssdi['lcd_type'] == lcd_type) &
                    (success_rate_by_ssdi['lds_type'] == lds_type)
                ]
                
                if not combo_data.empty:
                    combo_data = combo_data.sort_values('target_SSDI')
                    rates = combo_data['success'].values
                    rates = np.where(pd.isna(rates), 0, rates)
                    
                    line, = ax.plot(combo_data['target_SSDI'], rates,
                                  color=lcd_colors[lcd_type],
                                  linewidth=lds_linewidths[lds_type] + 1,
                                  linestyle=lds_linestyles[lds_type],
                                  marker=lds_markers[lds_type],
                                  markersize=10,
                                  markeredgecolor='white',
                                  markeredgewidth=1.5,
                                  label=f'{lcd_type}-LCD × {lds_type}-LDS',
                                  alpha=0.9,
                                  zorder=3)
                    
                    for i, (ssdi, rate) in enumerate(zip(combo_data['target_SSDI'], rates)):
                        if i == 0 or i == len(rates) - 1 or rate == max(rates) or rate == min(rates):
                            ax.annotate(f'{rate:.1f}%',
                                      xy=(ssdi, rate),
                                      xytext=(0, 10 if i % 2 == 0 else -15),
                                      textcoords='offset points',
                                      ha='center',
                                      fontsize=9,
                                      fontweight='bold',
                                      color=lcd_colors[lcd_type],
                                      bbox=dict(boxstyle='round,pad=0.3',
                                                facecolor='white',
                                                alpha=0.8,
                                                edgecolor=lcd_colors[lcd_type]),
                                      arrowprops=dict(arrowstyle='->',
                                                    color=lcd_colors[lcd_type],
                                                    alpha=0.6),
                                      zorder=4)
        
        ax.set_xlabel('Target SSDI Value', fontsize=16, fontweight='bold')
        ax.set_ylabel('Success Rate (%)', fontsize=16, fontweight='bold')
        ax.set_title('Detailed Success Rate Analysis: Nine Mechanism Combinations\nColor = LCD Mechanism | Line Width/Style = LDS Mechanism',
                    fontsize=18, fontweight='bold', pad=25)
        
        ax.set_xlim(min(ssdi_values) - 0.03, max(ssdi_values) + 0.03)
        ax.set_ylim(-5, 105)
        ax.set_xticks(ssdi_values)
        ax.set_xticklabels([f'{x:.2f}' for x in ssdi_values], fontsize=12)
        ax.set_yticks(np.arange(0, 101, 10))
        ax.set_yticklabels([f'{y}%' for y in np.arange(0, 101, 10)], fontsize=12)
        
        ax.grid(True, which='major', alpha=0.3, linestyle='-', linewidth=0.8)
        ax.grid(True, which='minor', alpha=0.2, linestyle=':', linewidth=0.5)
        ax.minorticks_on()
        
        lcd_patches = [Patch(color=color, label=f'{lcd_type}-LCD')
                      for lcd_type, color in lcd_colors.items()]
        
        lds_lines = [Line2D([0], [0], color='black',
                            lw=lds_linewidths[lds_type],
                            linestyle=lds_linestyles[lds_type],
                            label=f'{lds_type}-LDS')
                    for lds_type in ['client', 'special', 'lowrank']]
        
        legend1 = ax.legend(handles=lcd_patches, loc='upper left',
                          title='LCD Mechanism (Color)',
                          title_fontsize=12, fontsize=11,
                          framealpha=0.95, ncol=1)
        ax.add_artist(legend1)
        
        legend2 = ax.legend(handles=lds_lines, loc='upper right',
                          title='LDS Mechanism (Line Width/Style)',
                          title_fontsize=12, fontsize=11,
                          framealpha=0.95, ncol=1)
        
        overall_success_rate = combined_df['success'].mean() * 100
        summary_text = f"""
        Statistical Summary:
        • Overall Success Rate: {overall_success_rate:.1f}%
        • Best SSDI Range: {success_rate_by_ssdi.groupby('target_SSDI')['success'].mean().idxmax():.2f}
        • Worst SSDI Range: {success_rate_by_ssdi.groupby('target_SSDI')['success'].mean().idxmin():.2f}
        • Number of Samples: {len(combined_df):,}
        • Successful Trials: {combined_df['success'].sum():,}
        """
        ax.text(0.02, 0.02, summary_text, transform=ax.transAxes,
              fontsize=10, verticalalignment='bottom',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        fig_path = os.path.join(output_dir, 'nine_line_chart_detailed.png')
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  已保存: {fig_path}")
    
    print("\n" + "="*80)
    print("所有图表生成完成!")
    print(f"图表保存到: {output_dir}")
    print("="*80)

# ============================ 使用示例 ============================

def main():
    """
    使用示例
    """
    # 示例1: 运行generate_9_methods_and_analyse后使用
    """
    combined_df, detailed_stats_df, mechanism_stats_df, main_out_dir = generate_9_methods_and_analyse(
        C_list=[10],
        K_list=[20],
        N_list=[60000],
        SSDI_list=[0.3, 0.5, 0.7],
        repeats=30,
        alpha=2.0,
        beta=1.0,
        ssdi_error=0.02,
        seed=42,
        max_iters=100,
        repeats_factor=2
    )
    
    # 生成所有统计
    summary_stats = generate_statistics(
        (combined_df, detailed_stats_df, mechanism_stats_df, main_out_dir),
        stats_to_generate=["A", "B", "C", "D", "E", "F", "G"]
    )
    
    # 生成部分图表
    generate_plots(
        (combined_df, detailed_stats_df, mechanism_stats_df, main_out_dir),
        plots_to_generate=[1, 2, 3, 4, 5, 7]
    )
    """
    
    # 示例2: 从已有目录读取数据并生成统计和图表
    """
    main_out_dir = "./SSDI_Results_20240101_120000"
    
    # 生成所有统计
    summary_stats = generate_statistics(
        main_out_dir,
        stats_to_generate=["A", "B", "C", "D", "E", "F"]
    )
    
    # 生成所有图表
    generate_plots(
        main_out_dir,
        plots_to_generate=list(range(1, 11))
    )
    """
    
    print("请参考函数文档和示例代码使用本模块")

if __name__ == "__main__":
    main()




def _draw_lcd_lds_reference(ax, arc_radii=(0.2, 0.4, 0.6, 0.8, 1.0)):
    theta = np.linspace(0.0, np.pi / 2.0, 400)
    for r in arc_radii:
        x = r * np.sin(theta)
        y = r * np.cos(theta)
        ax.plot(x, y, linestyle='--', linewidth=0.85, color='0.78', zorder=0)
        label_theta = np.deg2rad(78)
        lx = r * np.sin(label_theta)
        ly = r * np.cos(label_theta)
        ax.text(lx, ly, f'{r:.1f}', fontsize=8, color='0.45', ha='left', va='bottom')
    ax.plot([0, 1], [0, 0], linestyle=':', linewidth=0.8, color='0.84', zorder=0)
    ax.plot([0, 0], [0, 1], linestyle=':', linewidth=0.8, color='0.84', zorder=0)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.10, linestyle='--')


def _plot_bias_specific_scatter_grids(combined_df: pd.DataFrame, output_dir: str):
    required = {"structure_bias", "target_SSDI", "lcd_type", "lds_type", "LCD", "LDS", "success"}
    if combined_df is None or combined_df.empty or not required.issubset(set(combined_df.columns)):
        return []

    plot_df = combined_df.dropna(
        subset=["structure_bias", "target_SSDI", "lcd_type", "lds_type", "LCD", "LDS", "success"]
    ).copy()
    if plot_df.empty:
        return []

    unique_bias = sorted(plot_df["structure_bias"].astype(float).unique())
    unique_ssdi = sorted(plot_df["target_SSDI"].astype(float).unique())
    if len(unique_bias) <= 0:
        return []

    lcd_order = ["client", "class", "joint"]
    lds_order = ["client", "special", "lowrank"]
    saved = []

    if len(unique_ssdi) > 1:
        norm = mpl.colors.Normalize(vmin=min(unique_ssdi), vmax=max(unique_ssdi))
    else:
        norm = None
    cmap = plt.cm.viridis

    for bias in unique_bias:
        bias_data = plot_df[np.isclose(plot_df["structure_bias"].astype(float), float(bias))].copy()
        if bias_data.empty:
            continue

        fig, axes = plt.subplots(3, 3, figsize=(14.8, 13.6), sharex=True, sharey=True)

        for i, lcd in enumerate(lcd_order):
            for j, lds in enumerate(lds_order):
                ax = axes[i, j]
                _draw_lcd_lds_reference(ax)

                mech = bias_data[
                    (bias_data["lcd_type"].astype(str) == lcd) &
                    (bias_data["lds_type"].astype(str) == lds)
                ].copy()

                if not mech.empty:
                    success_mask = mech["success"].astype(bool).to_numpy()
                    failure_mask = ~success_mask

                    if norm is not None:
                        colors = cmap(norm(mech["target_SSDI"].astype(float).to_numpy()))
                    else:
                        colors = np.array([cmap(0.5)] * len(mech))

                    if success_mask.any():
                        ax.scatter(
                            mech.loc[success_mask, "LCD"],
                            mech.loc[success_mask, "LDS"],
                            c=colors[success_mask],
                            s=34,
                            marker='o',
                            edgecolors='black',
                            linewidths=0.45,
                            alpha=0.95,
                            zorder=3,
                        )
                    if failure_mask.any():
                        ax.scatter(
                            mech.loc[failure_mask, "LCD"],
                            mech.loc[failure_mask, "LDS"],
                            facecolors='none',
                            edgecolors=colors[failure_mask],
                            s=38,
                            marker='o',
                            linewidths=1.0,
                            alpha=0.95,
                            zorder=2,
                        )

                ax.set_title(f"{lcd} × {lds}", fontsize=11)
                if i == 2:
                    ax.set_xlabel("LCD", fontsize=11)
                if j == 0:
                    ax.set_ylabel("LDS", fontsize=11)
                ax.set_xlim(0, 1.02)
                ax.set_ylim(0, 1.02)
                ax.set_aspect('equal', adjustable='box')

        if norm is not None:
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes, location='right', fraction=0.026, pad=0.045)
            cbar.set_label('Target SSDI', fontsize=11)

        legend_elements = [
            Line2D([0], [0], marker='o', color='black', markerfacecolor='0.55',
                   markeredgecolor='black', markersize=7, linewidth=0, label='Success (filled)'),
            Line2D([0], [0], marker='o', color='black', markerfacecolor='none',
                   markeredgecolor='black', markersize=7, linewidth=0, label='Failure (hollow)'),
        ]
        fig.legend(
            handles=legend_elements,
            loc='upper center',
            bbox_to_anchor=(0.48, 0.985),
            ncol=2,
            frameon=False,
            fontsize=10
        )
        fig.suptitle(f'Structured scatter by mechanism | structure_bias = {bias:.3f}', y=0.992, fontsize=14)

        path = os.path.join(output_dir, f'structured_bias_grid_{bias:+.3f}.png'.replace('+', 'p').replace('-', 'm'))
        _finalize_structured_grid_figure(fig, path, right=0.87, top=0.93, wspace=0.16, hspace=0.18)
        saved.append(path)

    return saved


def _plot_ssdi_specific_scatter_with_bias_colors(combined_df: pd.DataFrame, output_dir: str):
    required = {"structure_bias", "target_SSDI", "lcd_type", "lds_type", "LCD", "LDS", "success"}
    if combined_df is None or combined_df.empty or not required.issubset(set(combined_df.columns)):
        return []

    plot_df = combined_df.dropna(
        subset=["structure_bias", "target_SSDI", "lcd_type", "lds_type", "LCD", "LDS", "success"]
    ).copy()
    if plot_df.empty:
        return []

    unique_ssdi = sorted(plot_df["target_SSDI"].astype(float).unique())
    if len(unique_ssdi) <= 0:
        return []

    lcd_order = ["client", "class", "joint"]
    lds_order = ["client", "special", "lowrank"]
    cmap = plt.cm.coolwarm
    norm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)
    saved = []

    for ssdi_val in unique_ssdi:
        ssdi_data = plot_df[np.isclose(plot_df["target_SSDI"].astype(float), float(ssdi_val))].copy()
        if ssdi_data.empty:
            continue

        fig, axes = plt.subplots(3, 3, figsize=(14.8, 13.6), sharex=True, sharey=True)

        for i, lcd in enumerate(lcd_order):
            for j, lds in enumerate(lds_order):
                ax = axes[i, j]
                _draw_lcd_lds_reference(ax)

                mech = ssdi_data[
                    (ssdi_data["lcd_type"].astype(str) == lcd) &
                    (ssdi_data["lds_type"].astype(str) == lds)
                ].copy()

                if not mech.empty:
                    success_mask = mech["success"].astype(bool).to_numpy()
                    failure_mask = ~success_mask
                    colors = cmap(norm(mech["structure_bias"].astype(float).to_numpy()))

                    if success_mask.any():
                        ax.scatter(
                            mech.loc[success_mask, "LCD"],
                            mech.loc[success_mask, "LDS"],
                            c=colors[success_mask],
                            s=34,
                            marker='o',
                            edgecolors='black',
                            linewidths=0.45,
                            alpha=0.95,
                            zorder=3,
                        )
                    if failure_mask.any():
                        ax.scatter(
                            mech.loc[failure_mask, "LCD"],
                            mech.loc[failure_mask, "LDS"],
                            facecolors='none',
                            edgecolors=colors[failure_mask],
                            s=38,
                            marker='o',
                            linewidths=1.0,
                            alpha=0.95,
                            zorder=2,
                        )

                ax.set_title(f"{lcd} × {lds}", fontsize=11)
                if i == 2:
                    ax.set_xlabel("LCD", fontsize=11)
                if j == 0:
                    ax.set_ylabel("LDS", fontsize=11)
                ax.set_xlim(0, 1.02)
                ax.set_ylim(0, 1.02)
                ax.set_aspect('equal', adjustable='box')

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, location='right', fraction=0.026, pad=0.045)
        cbar.set_label('structure_bias (-1 = LDS, 0 = balance, 1 = LCD)', fontsize=11)

        legend_elements = [
            Line2D([0], [0], marker='o', color='black', markerfacecolor='0.55',
                   markeredgecolor='black', markersize=7, linewidth=0, label='Success (filled)'),
            Line2D([0], [0], marker='o', color='black', markerfacecolor='none',
                   markeredgecolor='black', markersize=7, linewidth=0, label='Failure (hollow)'),
        ]
        fig.legend(
            handles=legend_elements,
            loc='upper center',
            bbox_to_anchor=(0.48, 0.985),
            ncol=2,
            frameon=False,
            fontsize=10
        )
        fig.suptitle(f'Structured scatter by mechanism | target_SSDI = {float(ssdi_val):.3f}', y=0.992, fontsize=14)

        path = os.path.join(output_dir, f'structured_ssdi_grid_{float(ssdi_val):.3f}.png'.replace('.', 'p'))
        _finalize_structured_grid_figure(fig, path, right=0.87, top=0.93, wspace=0.16, hspace=0.18)
        saved.append(path)

    return saved


def _finalize_structured_grid_figure(
    fig,
    path: str,
    *,
    dpi: int = 220,
    right: float = 0.88,
    top: float = 0.93,
    left: float = 0.06,
    bottom: float = 0.06,
    wspace: float = 0.18,
    hspace: float = 0.20,
):
    """
    Finalize structured 3x3 scatter figures in a stable way.

    设计原则：
    - 不与 fig.colorbar(..., ax=axes) 的自动布局打架；
    - 通过 subplots_adjust 显式给右侧 colorbar 预留空间；
    - 不再叠加 bbox_inches='tight' 二次裁剪。
    """
    fig.subplots_adjust(
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        wspace=wspace,
        hspace=hspace,
    )
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# 2026-03 structured continuous-bias overlay support

def _plot_structured_bias_overlay(combined_df: pd.DataFrame, output_dir: str):
    required = {"structure_bias", "lcd_type", "lds_type", "LCD", "LDS", "success"}
    if combined_df is None or combined_df.empty or not required.issubset(set(combined_df.columns)):
        return None

    plot_df = combined_df.dropna(subset=["structure_bias", "lcd_type", "lds_type", "LCD", "LDS", "success"]).copy()
    if plot_df.empty:
        return None

    lcd_order = ["client", "class", "joint"]
    lds_order = ["client", "special", "lowrank"]

    fig, axes = plt.subplots(3, 3, figsize=(14.8, 13.6), sharex=True, sharey=True)
    cmap = plt.cm.coolwarm
    norm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)

    for i, lcd in enumerate(lcd_order):
        for j, lds in enumerate(lds_order):
            ax = axes[i, j]
            mech = plot_df[
                (plot_df["lcd_type"].astype(str) == lcd) &
                (plot_df["lds_type"].astype(str) == lds)
            ].copy()

            _draw_lcd_lds_reference(ax)

            if not mech.empty:
                success_mask = mech["success"].astype(bool).to_numpy()
                failure_mask = ~success_mask
                colors = cmap(norm(mech["structure_bias"].astype(float).to_numpy()))

                if success_mask.any():
                    ax.scatter(
                        mech.loc[success_mask, "LCD"],
                        mech.loc[success_mask, "LDS"],
                        c=colors[success_mask],
                        s=34,
                        marker='o',
                        edgecolors='black',
                        linewidths=0.45,
                        alpha=0.95,
                        zorder=3,
                    )
                if failure_mask.any():
                    ax.scatter(
                        mech.loc[failure_mask, "LCD"],
                        mech.loc[failure_mask, "LDS"],
                        facecolors='none',
                        edgecolors=colors[failure_mask],
                        s=38,
                        marker='o',
                        linewidths=1.0,
                        alpha=0.95,
                        zorder=2,
                    )

            ax.set_title(f"{lcd} × {lds}", fontsize=11)
            if i == 2:
                ax.set_xlabel("LCD", fontsize=11)
            if j == 0:
                ax.set_ylabel("LDS", fontsize=11)
            ax.set_xlim(0, 1.02)
            ax.set_ylim(0, 1.02)
            ax.set_aspect('equal', adjustable='box')

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location='right', fraction=0.026, pad=0.045)
    cbar.set_label('structure_bias (-1 = LDS, 0 = balance, 1 = LCD)', fontsize=11)

    legend_elements = [
        Line2D([0], [0], marker='o', color='black', markerfacecolor='0.55',
               markeredgecolor='black', markersize=7, linewidth=0, label='Success (filled)'),
        Line2D([0], [0], marker='o', color='black', markerfacecolor='none',
               markeredgecolor='black', markersize=7, linewidth=0, label='Failure (hollow)'),
    ]
    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.48, 0.985),
        ncol=2,
        frameon=False,
        fontsize=10
    )

    fig.suptitle('Structured overlay: continuous structure bias', y=0.992, fontsize=14)
    path = os.path.join(output_dir, 'structured_bias_overlay.png')
    _finalize_structured_grid_figure(fig, path, right=0.87, top=0.93, wspace=0.16, hspace=0.18)
    return path




def _plot_phase_vs_target_ssdi(combined_df: pd.DataFrame, output_dir: str):
    if combined_df is None or combined_df.empty or 'phase' not in combined_df.columns or 'target_SSDI' not in combined_df.columns:
        return None
    plot_df = combined_df.dropna(subset=['target_SSDI']).copy()
    if plot_df.empty:
        return None
    order = ['exact_zero', 'near_zero', 'normal', 'near_one', 'exact_one']
    phase_to_y = {p: i for i, p in enumerate(order)}
    phases = plot_df['phase'].fillna('normal').astype(str)
    y = phases.map(lambda x: phase_to_y.get(x, phase_to_y['normal']))
    x = pd.to_numeric(plot_df['target_SSDI'], errors='coerce')
    success = pd.to_numeric(plot_df.get('success', 1), errors='coerce').fillna(0).astype(float)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.scatter(x[success > 0.5], y[success > 0.5] + 0.03, alpha=0.7, s=28, label='success')
    ax.scatter(x[success <= 0.5], y[success <= 0.5] - 0.03, alpha=0.7, s=28, marker='x', label='best-effort')
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel('Target SSDI')
    ax.set_title('Generation phase by target SSDI')
    ax.grid(True, alpha=0.25)
    ax.legend()
    path = os.path.join(output_dir, 'phase_vs_target_ssdi.png')
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return path


def _plot_near_extremes_lcd_lds(combined_df: pd.DataFrame, output_dir: str):
    if combined_df is None or combined_df.empty or 'phase' not in combined_df.columns:
        return []
    paths = []
    for phase in ['near_zero', 'near_one', 'exact_zero', 'exact_one']:
        sub = combined_df[combined_df['phase'] == phase].copy()
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(5.6, 5.2))
        success = pd.to_numeric(sub.get('success', 1), errors='coerce').fillna(0).astype(float) > 0.5
        ax.scatter(sub.loc[success, 'LCD'], sub.loc[success, 'LDS'], s=34, alpha=0.7, label='success')
        ax.scatter(sub.loc[~success, 'LCD'], sub.loc[~success, 'LDS'], s=34, alpha=0.7, marker='x', label='best-effort')
        theta = np.linspace(0, np.pi/2, 200)
        ax.plot(np.sin(theta), np.cos(theta), linestyle='--', linewidth=1.0, alpha=0.6)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('LCD')
        ax.set_ylabel('LDS')
        ax.set_title(f'LCD-LDS points: {phase}')
        ax.grid(True, alpha=0.25)
        ax.legend()
        path = os.path.join(output_dir, f'lcd_lds_{phase}.png')
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches='tight')
        plt.close(fig)
        paths.append(path)
    return paths

def generate_plots(data_source, plots_to_generate=None, output_dir=None, show: bool = True):
    """Generate plots and display key figures in notebook/Colab.

    Adds plots 22-24 by default. For structured continuous-bias batches, also emits
    one overlay figure with continuous color, success/failure legend, and SSDI arcs.
    """
    if plots_to_generate is None:
        plots_to_generate = list(range(1, 11)) + [22, 23, 24]
    _generate_plots_legacy(data_source, plots_to_generate=plots_to_generate, output_dir=output_dir)

    if isinstance(data_source, tuple):
        combined_df, _, _, main_out_dir = data_source
    else:
        main_out_dir = data_source
        combined_df, _, _ = load_data_from_dir(main_out_dir)
    if output_dir is None:
        output_dir = os.path.join(main_out_dir, 'fig')

    structured_extra = []
    extra_paths = []
    if combined_df is not None and 'structure_bias' in combined_df.columns:
        overlay_path = _plot_structured_bias_overlay(combined_df, output_dir)
        if overlay_path:
            structured_extra.append(overlay_path)
        structured_extra.extend(_plot_bias_specific_scatter_grids(combined_df, output_dir))
        structured_extra.extend(_plot_ssdi_specific_scatter_with_bias_colors(combined_df, output_dir))

    extra_paths.extend([p for p in [_plot_phase_vs_target_ssdi(combined_df, output_dir)] if p])
    extra_paths.extend(_plot_near_extremes_lcd_lds(combined_df, output_dir))

    if show:
        for name in [
            'scatter_all_success.png',
            'scatter_all_samples.png',
            'subplot_matrix_all_ssdi.png',
            'mechanism_success_heatmap.png',
        ]:
            _display_image_if_exists(os.path.join(output_dir, name))
        for path in structured_extra + extra_paths:
            _display_image_if_exists(path)

    return {
        'output_dir': output_dir,
        'scatter_all_success': os.path.join(output_dir, 'scatter_all_success.png'),
        'scatter_all_samples': os.path.join(output_dir, 'scatter_all_samples.png'),
        'subplot_matrix_all_ssdi': os.path.join(output_dir, 'subplot_matrix_all_ssdi.png'),
        'mechanism_success_heatmap': os.path.join(output_dir, 'mechanism_success_heatmap.png'),
        'structured_extra_plots': structured_extra,
        'phase_plots': extra_paths,
    }


# ============================ User-requested plotting patch v3 ============================
from matplotlib.colors import LogNorm, PowerNorm
from matplotlib import cm
from IPython.display import Image as _IPImage, display as _ip_display


def _display_image_if_exists(path: str):
    if path and os.path.exists(path):
        try:
            _ip_display(_IPImage(filename=path))
        except Exception:
            pass


def plot_single_matrix_distribution(
    df: pd.DataFrame,
    details: dict | None = None,
    output_dir: str | None = None,
    prefix: str | None = None,
    show: bool = True,
    dpi: int = 160,
):
    """
    Plot a single matrix with strong low-count contrast.

    Design:
    - 0 -> pure white
    - 1 and other small counts are clearly distinguishable from 0
    - contrast among small positive values is emphasized
    - very large values are visually compressed
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    if output_dir is None:
        output_dir = (details or {}).get("output_dir", os.getcwd())
    os.makedirs(output_dir, exist_ok=True)

    if prefix is None:
        prefix = "single_matrix"

    arr = df.to_numpy(dtype=float)
    saved = {}

    def _save_show(fig, path):
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    cmap = cm.get_cmap("YlOrRd").copy()
    cmap.set_bad(color="white")

    # 1) count heatmap
    positive = arr[arr > 0]
    if positive.size > 0:
        vmax = float(np.percentile(positive, 99.5))
        vmax = max(vmax, 2.0)
        count_display = np.where(arr > 0, arr, np.nan)
        norm = LogNorm(vmin=1.0, vmax=vmax)
    else:
        count_display = np.full_like(arr, np.nan)
        norm = None
        vmax = 1.0

    fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=dpi)
    im1 = ax1.imshow(count_display, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax1.set_title("Count Heatmap (zero = white, low counts emphasized)")
    ax1.set_xlabel("Label / Class")
    ax1.set_ylabel("Client")

    if positive.size > 0:
        cbar1 = fig1.colorbar(im1, ax=ax1, shrink=0.9)
        tick_candidates = np.array([1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000], dtype=float)
        ticks = tick_candidates[(tick_candidates >= 1) & (tick_candidates <= vmax)]
        if ticks.size == 0:
            ticks = np.array([1.0, vmax])
        cbar1.set_ticks(ticks)
        cbar1.set_ticklabels([str(int(t)) if float(t).is_integer() else f"{t:.1f}" for t in ticks])
        cbar1.set_label("Count")

    p1 = os.path.join(output_dir, f"{prefix}_heatmap.png")
    _save_show(fig1, p1)
    saved["heatmap"] = p1

    # 2) row-normalized heatmap
    row_sums = arr.sum(axis=1, keepdims=True)
    row_norm = np.divide(arr, row_sums, out=np.zeros_like(arr, dtype=float), where=row_sums > 0)
    row_norm_display = np.where(row_norm > 0, row_norm, np.nan)
    finite_vals = row_norm_display[np.isfinite(row_norm_display)]
    vmax_row = float(np.percentile(finite_vals, 99.5)) if finite_vals.size else 1.0
    vmax_row = max(vmax_row, 1e-3)

    fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=dpi)
    im2 = ax2.imshow(
        row_norm_display,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=PowerNorm(gamma=0.45, vmin=1e-6, vmax=vmax_row),
    )
    ax2.set_title("Row-normalized Heatmap (small positive mass emphasized)")
    ax2.set_xlabel("Label / Class")
    ax2.set_ylabel("Client")
    cbar2 = fig2.colorbar(im2, ax=ax2, shrink=0.9)
    cbar2.set_label("Within-client proportion")

    p2 = os.path.join(output_dir, f"{prefix}_row_normalized_heatmap.png")
    _save_show(fig2, p2)
    saved["row_normalized_heatmap"] = p2

    # 3) support mask
    fig3, ax3 = plt.subplots(figsize=(10, 6), dpi=dpi)
    ax3.imshow((arr > 0).astype(float), aspect="auto", interpolation="nearest", cmap="Greys")
    ax3.set_title("Support Mask (white = missing, black = present)")
    ax3.set_xlabel("Label / Class")
    ax3.set_ylabel("Client")

    p3 = os.path.join(output_dir, f"{prefix}_support_mask.png")
    _save_show(fig3, p3)
    saved["support_mask"] = p3

    # 4) marginals
    fig4, axes = plt.subplots(2, 1, figsize=(10, 7), dpi=dpi)
    axes[0].bar(np.arange(arr.shape[1]), arr.sum(axis=0))
    axes[0].set_title("Client Marginal Sizes")
    axes[0].set_xlabel("Client")
    axes[0].set_ylabel("Samples")

    axes[1].bar(np.arange(arr.shape[0]), arr.sum(axis=1))
    axes[1].set_title("Class Marginal Sizes")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Samples")

    p4 = os.path.join(output_dir, f"{prefix}_marginals.png")
    _save_show(fig4, p4)
    saved["marginals"] = p4

    return saved

