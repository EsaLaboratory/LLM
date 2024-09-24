import argparse
from datetime import datetime
import numpy as np
from home_management import Market, Model, tests


def main():
    "Creation of the optim command"
    parser = argparse.ArgumentParser(
        description="Output an LLM's answer to a question on documents")
    parser.add_argument('--data_path',
                        metavar='Data path',
                        type=str,
                        default="../data/",
                        help="Data path")
    parser.add_argument('--dno',
                        metavar='dno',
                        type=str,
                        default='EDF',
                        help="DNO of the house")
    parser.add_argument('--N',
                        metavar='N',
                        type=int,
                        default=24,
                        help="Number of iterations")
    parser.add_argument('--EV',
                        metavar='EV',
                        type=int,
                        default=2,
                        help="Number of Electrical Vehicules")
    parser.add_argument('--date_start',
                        metavar='date_start',
                        type=str,
                        default="03/05/24 00:00:00.0",
                        help="Start date of simulation")
    parser.add_argument('--date_end',
                        metavar='date_end',
                        type=str,
                        default="05/05/24 00:00:00.0",
                        help="End date of simulation")
    parser.add_argument('--plot_path',
                        metavar='plot_path',
                        type=str,
                        default="../img/optim.pdf",
                        help="Path for plot output")
    parser.add_argument('--departure_time',
                        metavar='departure_time',
                        type=int,
                        default=8,
                        help="End of the simulation")
    parser.add_argument('--arrival_time',
                        metavar='arrival_time',
                        type=int,
                        default=18,
                        help="Start of the simulation")
    parser.add_argument('--Tmin',
                        metavar='Tmin',
                        type=float,
                        default=18,
                        help="Minimal temperature Kelvin")
    parser.add_argument('--Tmax',
                        metavar='Tmax',
                        type=float,
                        default=20,
                        help="Maximal temperature Kelvin")
    parser.add_argument('--plot_type',
                        metavar='plot_type',
                        type=str,
                        default="horizontal",
                        help="Plot type, horizontal or vertical")

    args = parser.parse_args()

    data_path = args.data_path
    dno = args.dno
    EV = args.EV
    date_start = datetime.strptime(args.date_start, "%d/%m/%y %H:%M:%S.%f")
    date_end = datetime.strptime(args.date_end, "%d/%m/%y %H:%M:%S.%f")
    N = (date_end - date_start).days * 24
    departure_time = args.departure_time
    arrival_time = args.arrival_time
    Tmin = args.Tmin
    Tmax = args.Tmax
    plot_path = args.plot_path
    plot_type = args.plot_type

    market_kwargs = {
        'data_path': data_path,
        'dno': dno,
        'N': N
    }
    market = Market(**market_kwargs)

    model_kwargs = {
        'data_path': data_path,
        'EV': EV,
        'date_start': date_start,
        'date_end': date_end,
        'departure_time': departure_time,
        'arrival_time': arrival_time,
        'Tmin': Tmin,
        'Tmax': Tmax,
        'market': market,
        'plot_path': plot_path,
    }
    user = Model(**model_kwargs)
    tests(user)
    user.optimization()
    if plot_type == 'vertical':
        user.plot_vertical()
    elif plot_type == 'separately':
        user.plot_separately()
    else:
        user.plot()


if __name__ == '__main__':
    main()
