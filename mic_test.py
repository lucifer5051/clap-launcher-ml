import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    print(f"Device {i}: {p.get_device_info_by_index(i)['name']}")
try:
    s = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
    print("Recording 2 seconds...")
    for _ in range(0, int(44100 / 1024 * 2)): s.read(1024)
    print("Microphone working!")
    s.close()
except Exception as e:
    print(f"Microphone test failed! Exact error:\n{e}")
p.terminate()
