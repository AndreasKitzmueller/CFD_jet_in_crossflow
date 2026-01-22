import plot_utils

def main(fontsize = 16, width = 1600, height = 450):
    fig, info = plot_utils.plot_polymesh_case(
            "case",
            unit="mm",
            crease_angle_deg=10.0,
            inlet_patch="inletMain",
            outlet_patch="outlet",
            jet_patch="jetInlet",
            label_shift_x_factor_big=0.75,
            label_shift_x_factor_jet=2,
            fontsize=fontsize,
            width=width,
            height=height
        )
    fig.show()