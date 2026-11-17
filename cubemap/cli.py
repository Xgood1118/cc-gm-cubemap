import argparse
import cubemap as m


def main():
    ap = argparse.ArgumentParser(description="立方体贴图接缝与过滤")
        ap.add_argument("--dir", required=True)
        ap.add_argument("--out", required=True)
    a = ap.parse_args()
        print("TODO")


if __name__ == "__main__":
    main()
