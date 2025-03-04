import psutil

print("Antal logiska processorer:", psutil.cpu_count(logical=True))
print("Antal fysiska kärnor:", psutil.cpu_count(logical=False))
