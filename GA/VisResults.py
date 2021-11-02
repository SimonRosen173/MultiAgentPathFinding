import pickle
from Visualisations.Vis import VisGrid


def main():
    with open("local/pop_2000.pkl", "rb") as f:
        data = pickle.load(f)

    best_data = sorted(data, key=lambda x: x[1], reverse=True)[:5]
    for i, curr_data in sorted(best_data):
        pass

    # best_grid, fit = max(data, key=lambda x: x[1])
    # print("Fitness", fit)
    # vis = VisGrid(best_grid, (800, 400), 25, tick_time=0.2)
    # vis.save_to_png(f"local/imgs/best")
    # vis.window.close()
    # print(best)


if __name__ == "__main__":
    main()
