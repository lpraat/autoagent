def log(s, files):
    for file in files:
        with open(file, mode='a', encoding='utf-8') as f:
            f.write(s)

    print(s, end="")

fancy_float = lambda x: f"{x:.5f}"

def get_hour_min_sec(ts):
    hours = ts // 3600
    minutes = (ts % 3600) // 60
    seconds = ts % 60
    return hours, minutes, seconds

def format_hms(ts):
    h, m, s = get_hour_min_sec(ts)
    return f"{int(h)} hours, {int(m)} minutes, {int(s)} seconds"
