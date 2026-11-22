import pandas as pd

df = pd.DataFrame({
    "timestamp": [
        "2026-03-02 08:55", "2026-03-02 09:12", "2026-03-02 09:40", "2026-03-02 10:03",
        "2026-03-02 10:40", "2026-03-02 11:15", "2026-03-03 08:50", "2026-03-03 09:20",
        "2026-03-03 09:45", "2026-03-03 10:30", "2026-03-03 11:05", "2026-03-03 14:10",
    ],
    "person": [
        "worker_B", "worker_A", "worker_C", "worker_A",
        None, "worker_B", "worker_A", "worker_C",
        "worker_A", "worker_B", None, "worker_C",
    ],
    "event": [
        "enter_zone", "no_helmet", "no_helmet", "no_helmet",
        "no_helmet", "enter_zone", "no_helmet", "leave_zone",
        "enter_zone", "no_helmet", "no_helmet", "no_helmet",
    ],
})
print(f"1. 原始資料：{len(df)}筆")
print(df.dtypes)

print("---")
print("2. 各欄缺值數：")
print(df.isna().sum())
df_clean = df.dropna()
print(f"刪掉缺值後： {len(df_clean)} 筆")

print("---")

nh_df = df_clean[df_clean["event"] == "no_helmet"]

print("3. 未戴安全帽事件：")
print(nh_df)
print("---")

counts = nh_df.groupby("person").size()
print("4. 每人未戴安全帽次數：")
print(counts)
print("---")

summary = counts.reset_index(name="no_helmet_count").to_dict(orient="records")
print("5. 給 LLM 的摘要：")
print(summary)