
    def compute_features_with(self, out_days,time_range,tsfresh_filename):
        visible_date=None
        vpv_noise_threshold = 250
        ipv_threshold = 5000
        vpv_default = 1050
        vpv_spike_amplitude = [0.8, 0.9, 1.0, 1.1, 1.2]
        vpv_sag_amplitude = [0.05, 0.1]
        vac_default = 220
        vac_spike_amplitude = [1.1, 1.2, 1.3]
        vac_sag_amplitude = [0.8, 0.9, 1.0]        

        df = self.df.with_columns(
        pl.col('createtime').str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
        )
        #df = df.with_columns(pl.col('errorcode').cast(pl.Float64))
        df = df.with_columns(
        pl.col('errorcode').cast(pl.Utf8)  # 将列转换为字符串类型
        )


        sn_list=[]
        time_list=[]

        groups = df.partition_by("sn")
        feature_list = []
        for group_df in groups:
            sn = group_df['sn'][0]
            sn_list.append(sn)
            group_df = group_df.sort('createtime')
            # 根据date判断可利用的最后一天数据,然后剔除out_day天的数据
            # 最后仅保留time_range天的数据用于特征生成
            if visible_date:
                    #print("===========")
                    #print(visible_date)
                try:
                    visible_date = visible_date.date()
                except:
                    visible_date
                last_date = min(group_df['createtime'].dt.date().max(), visible_date) - timedelta(days=out_days-1)
            else:
                last_date = group_df['createtime'].dt.date().max() - timedelta(days=out_days-1)
            start_date = last_date - timedelta(days=time_range)
            group_df_filtered = group_df.filter((pl.col('createtime') > start_date) &
                                                (pl.col('createtime') < last_date))
            # 构建日期列
            time_list.append(last_date)
            group_df_filtered = group_df_filtered.with_columns(
                pl.col('createtime').dt.date().alias("date")
            )

            # 构建error_code有无状态
            group_df_filtered = group_df_filtered.with_columns(
                (~pl.col("errorcode").is_in(["0"])).alias("error_status")
            )
            # 构建vac的横向平均
            group_df_filtered = group_df_filtered.with_columns(
                ((pl.col("vac1")+pl.col("vac2")+pl.col("vac3"))/3).alias("vac_horizontal_mean")
            )
            group_df_filtered = group_df_filtered.with_columns(
                (
                    ((pl.col(f"vac1")-pl.col("vac_horizontal_mean")).pow(2) + 
                    (pl.col(f"vac2")-pl.col("vac_horizontal_mean")).pow(2) + 
                    (pl.col(f"vac3")-pl.col("vac_horizontal_mean")).pow(2))/3
                ).alias("vac_horizontal_variance")
            )
            # 构建日度error_rate,on_rate,eday特征
            agg_expr = []
            agg_expr.append(pl.col("error_status").mean().alias('error_rate'))
            agg_expr.append((pl.col("errorcode")!="0").any().alias('error_occur'))
            agg_expr.append(pl.col("switchstatus").mean().alias('on_rate'))
            agg_expr.append(pl.col("vac_horizontal_variance").mean().alias('vac_horizontal_variance'))
            for i,j in [[1,2], [3,4], [5,6]]:
                agg_expr.append(pl.col(f"vac_vpv_ratio_{i}{j}").mean().alias(f"vac_vpv_ratio_{i}{j}"))
                agg_expr.append(pl.col(f"iac_ipv_ratio_{i}{j}").mean().alias(f"iac_ipv_ratio_{i}{j}"))
            agg_expr.append(pl.col('sn').first().alias('sn'))
            for i in range(1,7):
                agg_expr.append((pl.col(f"ipv{i}")>ipv_threshold).any().alias(f"ipv{i}_peaks"))
            for i in range(1,7):
                print(vpv_spike_amplitude,vpv_default)
                for amp in vpv_spike_amplitude:
                    agg_expr.append((pl.col(f"vpv{i}")>vpv_default*amp).sum().alias(f"vpv{i}_{amp}_spikes"))
                for amp in vpv_sag_amplitude:
                    agg_expr.append((pl.col(f"vpv{i}")<vpv_default*amp).sum().alias(f"vpv{i}_{amp}_sags"))
            for i in range(1,4):
                for amp in vac_spike_amplitude:
                    agg_expr.append((pl.col(f"vac{i}")>vac_default*amp).sum().alias(f"vac{i}_{amp}_spikes"))
                for amp in vac_sag_amplitude:
                    agg_expr.append((pl.col(f"vac{i}")<vac_default*amp).sum().alias(f"vac{i}_{amp}_sags"))
            iac_threshold = int(sn[2:4])*1000/220/1.732/3
            for i in range(1,4):agg_expr.append((pl.col(f"iac{i}")>iac_threshold).sum().alias(f"iac{i}_peaks"))
            print(group_df_filtered)
            print(group_df_filtered['date'])
            result_1 = group_df_filtered.group_by('date').agg(agg_expr)
            # 填充上所有日期的数据,若数据缺失用后一天的数据填充
            date_df = pl.DataFrame({'date':pl.date_range(start=start_date, end=last_date, interval="1d", closed="left", eager=True)})
            result_1 = date_df.join(result_1, on = 'date', how = 'left').sort('date')

            # 在剔除switch_status为0（交流断路）的情况下,生成日度vpv, vac特征
            target_cols = ([f"vac{i}" for i in range(1,4)]
                         + [f"vpv{i}" for i in range(1,7)]
                          )
            agg_expr = []
            for col in target_cols:
                agg_expr.append(pl.col(col).mean().alias(f'{col}_mean'))
                agg_expr.append(pl.col(col).var().alias(f'{col}_variance'))
            result_2 = group_df_filtered.filter(pl.col('switchstatus')==1).group_by('date').agg(agg_expr)

            # 在仅仅考虑11:00-15:00数据的情况下(确保取出来为满负荷工作状态的temperature)
            # 同时,也剔除switchstatus为0的数据
            # 生成temperature特征
            result_3 = (
                group_df_filtered
                .filter(pl.col('switchstatus')==1)
                .filter((pl.col('createtime').dt.hour() >= 11) & (pl.col('createtime').dt.hour() < 15))
                .group_by('date')
                .agg(pl.col('temperature').mean().alias('temperature_mean'))
                )

            # 从业务侧得知,早晚eday计算存在误差,仅考虑9:00-15:00的eday更准确
            result_4 = (
                group_df_filtered
                .filter((pl.col('createtime').dt.hour() >= 9) & (pl.col('createtime').dt.hour() < 15))
                .group_by('date')
                .agg([pl.col('eday').first().alias('eday_start'),
                      pl.col('eday').last().alias('eday_end'),
                      pl.col("switchstatus").mean().alias('on_rate_4_eday'),
                      pl.col("activepower").mean().alias('activepower_mean')])
                )

            # result_1, result_2, result_3结果合并
            # !!!TODO:注意fill_null的效果是否会影响结果
            result = result_1.join(result_2, on = 'date', how = 'left')
            result = result.join(result_3, on = 'date', how = 'left')
            result = result.join(result_4, on = 'date', how = 'left').sort('date').fill_null(strategy="backward").fill_null(0)

            def percentage_difference(data):
                """若分母为0,则替换成1,以消除inf
                """
                numerator = (data.diff()[1:]).to_numpy()
                denominator = (data[:-1]).to_numpy()

                output = np.divide(numerator, denominator, where=(denominator!=0))
                return output.tolist()

            # 产生vpv特征
            vpv_mean = {}
            vpv_variance = {}
            vpv_at_work = {}
            for i in range(1,7):
                vpv_at_work[i] = result[f"vpv{i}_mean"].max() < vpv_noise_threshold
                if result[f"vpv{i}_mean"].max() < vpv_noise_threshold:
                    vpv_mean[i] = [0]*(time_range-1)
                    vpv_variance[i] = [0]*(time_range-1)
                else:
                    vpv_mean[i] = percentage_difference(result[f"vpv{i}_mean"])
                    vpv_variance[i] = percentage_difference(result[f"vpv{i}_variance"])
            # 产生vac特征
            vac_mean = {}
            vac_variance = {}
            for i in range(1,4):
                vac_mean[i] = percentage_difference(result[f"vac{i}_mean"])
                vac_variance[i] = percentage_difference(result[f"vac{i}_variance"])
            # 产生温度特征
            temperature = result["temperature_mean"].to_list()
            temperature_diff = percentage_difference(result["temperature_mean"])
            temperature_vs_power = np.divide(result["temperature_mean"], result['activepower_mean'],
                                    where=(result['activepower_mean'].to_numpy()!=0)).to_list()
            # 产生eday特征
            eday = np.divide((result['eday_end'] - result['eday_start']), result['on_rate_4_eday'],
                             where=(result['on_rate_4_eday'].to_numpy()!=0)) 
            eday = percentage_difference(eday)
            # error_code比例特征
            error_occur = result["error_occur"].to_list()
            error_rate = result["error_rate"].to_list()
            # switch_status比例特征
            on_rate = result["on_rate"].to_list()
            # ipv peaks特征
            ipv_peaks = {}
            iac_peaks = {}
            vpv_spikes = {}
            vac_spikes = {}
            vpv_sags = {}
            vac_sags = {}
            for i in range(1,7):
                ipv_peaks[i] = result[f"ipv{i}_peaks"].to_list()
                for amp in vpv_spike_amplitude:
                    vpv_spikes[f"{i}_{amp}"] = result[f"vpv{i}_{amp}_spikes"].to_list()
                for amp in vpv_sag_amplitude:
                    vpv_sags[f"{i}_{amp}"] = result[f"vpv{i}_{amp}_sags"].to_list()
            for i in range(1,4):
                iac_peaks[i] = result[f"iac{i}_peaks"].to_list()
                for amp in vac_spike_amplitude:
                    vac_spikes[f"{i}_{amp}"] = result[f"vac{i}_{amp}_spikes"].to_list()
                for amp in vac_sag_amplitude:
                    vac_sags[f"{i}_{amp}"] = result[f"vac{i}_{amp}_sags"].to_list()
            vac_horizontal_variance = result["vac_horizontal_variance"].to_list()
            vac_vpv_ratio, iac_ipv_ratio = {}, {}
            for i,j in [[1,2], [3,4], [5,6]]:
                vac_vpv_ratio[f"{i}{j}"] = result[f"vac_vpv_ratio_{i}{j}"].to_list()
                iac_ipv_ratio[f"{i}{j}"] = result[f"iac_ipv_ratio_{i}{j}"].to_list()
            
            features = [sn]
            for i in range(1,7):features += vpv_mean[i]
            for i in range(1,7):features += vpv_variance[i]
            for i in range(1,4):features += vac_mean[i]
            for i in range(1,4):features += vac_variance[i]
            for i in range(1,7):features += ipv_peaks[i]
            for i in range(1,4):features += iac_peaks[i]
            for i in range(1,7):
                for amp in vpv_spike_amplitude:
                    features += vpv_spikes[f"{i}_{amp}"]
                for amp in vpv_sag_amplitude:
                    features += vpv_sags[f"{i}_{amp}"]
            for i in range(1,4):
                for amp in vac_spike_amplitude:
                    features += vac_spikes[f"{i}_{amp}"]
                for amp in vac_sag_amplitude:
                    features += vac_sags[f"{i}_{amp}"]
            features += vac_horizontal_variance + temperature + temperature_diff + temperature_vs_power
            features += eday + error_occur + error_rate + on_rate + list(vpv_at_work.values())
            feature_list.append(features)
    
        # 生成特征列名,并且拼接所有特征
        feature_names = (["sn"] + 
                         [f"vpv{i}_mean_day_{j}" for i in range(1,7) for j in range(1, time_range)] +
                         [f"vpv{i}_variance_day_{j}" for i in range(1,7) for j in range(1, time_range)] +
                         [f"vac{i}_mean_day_{j}" for i in range(1,4) for j in range(1, time_range)] +
                         [f"vac{i}_variance_day_{j}" for i in range(1,4) for j in range(1, time_range)] +
                         [f"ipv{i}_peaks_day_{j}" for i in range(1,7) for j in range(time_range)] +
                         [f"iac{i}_peaks_day_{j}" for i in range(1,4) for j in range(time_range)] +

                         [f"vpv{i}_{amp}_spikes_day_{j}" for i in range(1,7) for amp in vpv_spike_amplitude for j in range(time_range)] +
                         [f"vpv{i}_{amp}_sags_day_{j}" for i in range(1,7) for amp in vpv_sag_amplitude for j in range(time_range)] +
                         [f"vac{i}_{amp}_spikes_day_{j}" for i in range(1,4) for amp in vac_spike_amplitude for j in range(time_range)] +
                         [f"vac{i}_{amp}_sags_day_{j}" for i in range(1,4) for amp in vac_sag_amplitude for j in range(time_range)] +
                         
                         [f"vac_horizontal_variance_day_{j}" for j in range(time_range)] +
                         [f"temperature_day_{j}" for j in range(time_range)] +
                         [f"temperature_diff_day_{j}" for j in range(1, time_range)] +
                         [f"temperature_vs_power_day_{j}" for j in range(time_range)] +
                         [f"eday_day_{j}" for j in range(1, time_range)] +
                         [f"error_occur_day_{j}" for j in range(time_range)] +
                         [f"error_rate_day_{j}" for j in range(time_range)] +
                         [f"on_rate_day_{j}" for j in range(time_range)] +
                         [f"vpv_at_work_{i}" for i in range(1,7)] +
                         [f"vac_vpv_ratio_{i}{j}_day_{k}" for i,j in [[1,2], [3,4], [5,6]] for k in range(time_range)] +
                         [f"iac_ipv_ratio_{i}{j}_day_{k}" for i,j in [[1,2], [3,4], [5,6]] for k in range(time_range)]
                        )
        feature_df = pl.DataFrame(feature_list,
                                  schema=feature_names)

      
        
        path = "./data/tsfresh_feature"
        os.makedirs(path, exist_ok=True)
        current_date = datetime.now().strftime('%Y%m%d_%H%M')
        parquet_path = os.path.join(path, f'{current_date}_{tsfresh_filename}')
        #parquet_path='ok.csv'
        feature_df.write_parquet(parquet_path)
        #print(f"特征数据已保存为 {parquet_path}")
        #print('2222',feature_df)
        return feature_df,self.all_columns