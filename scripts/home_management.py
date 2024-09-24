import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import picos as pc
from datetime import timedelta, datetime
from typing import Callable, Any
import time
import json
import pandas as pd
import glob
import random


def timer(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Define a decorator for time measurement of a function call.

    Args:
        func: The function to decorate.

    Returns:
        An instance of a decorated function."""
    name = func.__name__

    def description(*args, **kwargs):
        arg_str = ', '.join(repr(arg) for arg in args)
        start = time.time()
        resultat = func(*args, **kwargs)
        end = time.time()
        time_func = str(timedelta(seconds=(end - start)))
        if kwargs is None:
            print(
                f"\nFunction {name}\nargs: {arg_str}\ndone in :{end - start}")
        else:
            key_word = ""
            for key in kwargs.keys():
                if len(repr(kwargs[key])) < 15:
                    key_word += ', ' + key + ": " + repr(kwargs[key])
                else:
                    key_word += ', ' + key + ": " + repr(type(kwargs[key]))
            print(
                f"\nFunction {name}\nargs {arg_str}\nkwargs\
                      {key_word}\ndone in :{time_func}")
        return resultat
    return description


class Market():
    """This class describes a basic electricity market.

    Attributes:
        prices: A list of electricity prices.
        dt: The time delta.
        N: The number of optimization point.
    """
    @timer
    def __init__(
        self,
        N: int,
        dno: str,
        data_path: str = '../data/',
    ):
        """Initialize a Market object.

        Args:
            data_path: Path to dno data
            dt: The time delta.
            N: The number of optimization point.
            dno: The dno selling the electricity.
            type: Tariff type.
        """
        with open(data_path + 'dno.json', 'r', encoding="utf8") as dno_file:
            DNO = json.load(dno_file)
            dno_file.close()
        self.prices = None
        self.N = N
        self.dt = 1
        self.day_price = DNO[dno]['day']
        self.night_price = DNO[dno]['night']
        self.solar_price = DNO[dno]['sge']
        self.dno = dno


class Model():
    """This class describes a basic user model.

    A Model instance able to optimize one user's consumption.

    Attributes:
        EV: The number of electric vehicules.
        departure_time: The leaving date of the vehicule (morning).
        arrival_time  : The arrival date of the vehicule (end of the day).
        Tmin: The minimum temparature in the house K.
        Tmax: The maximum temparature in the house K.
        market: The object containing electricity price evolution.
        EV_energy_goal: The amount of energy to fully charged car J.
        V: The volume of the house m^3.
        P: The pressure of the room kPa.
        max_power: The maximum usable power W.
        min_power: The minimum usable power W.
        cv: The heat capacity kJ/(kg.K).
        k: The thermal conductivity W/(m·K).
        s: The wall thickness m;
        R: The ideal gas constant kJ/(kg.K).
        T0: The initial temperature K.
        T_outside: The outside temperature K.
        power_t: The power consumption for house temperature W.
        power_e: The power consumption for EV charging W.
        power: The total power consumption W.
        energy: The electric vehicule energy J.
        temperature: The house temperature K.
        cost: The total cost of the used energy.
    """
    @timer
    def __init__(
        self,
        EV:  int,
        date_start: datetime,
        date_end: datetime,
        departure_time: int,
        arrival_time: int,
        Tmin: float,
        Tmax: float,
        market: Market,
        plot_path: str,
        data_path: str = '../data/'
    ):
        """Initialization of Model object.

        Args:
            EV: The number of electric vehicules.
            departure_time: The leaving date of the vehicule (morning).
            arrival_time  : The arrival date of the vehicule (end of the day).
            Tmin: The minimum temparature in the house K.
            Tmax: The maximum temparature in the house K.
            market: The object containing electricity price evolution.
            EV_energy_goal: The amount of energy to fully charged car J.
            V: The volume of the house m^3.
            P: The pressure of the room kPa.
            max_power: The maximum usable power W.
            min_power: The minimum usable power W.
            cv: The specific heat capacity kJ/(kg.K).
            k: The thermal conductivity W/(m·K).
            surface: Walls surface m^2;
            R: The ideal gas constant for dry air kJ/(kg.K).
            Rh: thermal resistance °C/kW
            Ch: thermal capacity kWh/°C
            U: u-values, conductance W/(m^2.K)
            T0: The initial temperature K.
            T_outside: The outside temperature K.
            power_t: The power consumption for house temperature kW.
            power_e: The power consumption for EV charging kW.
            power: The total power consumption kW.
            energy: The electric vehicule energy J.
            temperature: The house temperature K.
            cost: The total cost of the used energy."""
        self.EV = EV
        self.date_start = date_start
        self.date_end = date_end
        self.initial_time = date_start + timedelta(hours=departure_time)
        self.final_time = date_end + timedelta(hours=departure_time)
        self.arrival_time = arrival_time
        self.departure_time = departure_time
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.market = market
        self.EV_energy_goal = (144000000*self.EV) / 3600000  # 40 kwh per car
        self.max_power = 2.3 * (self.EV*(self.EV > 0) + 1*(self.EV == 0))
        self.min_power = 0
        self.Rh = 10
        self.Ch = 2
        self.alpha = 1 / (self.Rh * self.Ch)
        self.beta = 1 / self.Ch
        self.T0 = (Tmax+Tmin) / 2
        self.E0 = self.EV_energy_goal * \
            (np.random.random(self.market.N//24 + 1)*0.5 + 0.2)
        self.power_t = pc.RealVariable("power_t", self.market.N)
        self.power_e = pc.RealVariable("power_e", self.market.N)
        # From TC3 dataset
        HOUSE_CONSO_PATHS = [path[path.find("TC3"):]
                             for path in glob.glob(data_path + "TC3*")]
        random_path = [random.choice(HOUSE_CONSO_PATHS)
                       for i in range(self.market.N//24)]
        self.power_h = np.array([])
        try:
            for path in random_path:
                with open(data_path + path, "r") as fp:
                    self.power_h = np.concatenate(
                        [self.power_h, json.load(fp)])
                    fp.close()
        except Exception as e:
            print(HOUSE_CONSO_PATHS)
        self.energy = pc.RealVariable("energy", self.market.N+1)
        self.temperature = pc.RealVariable("temperature", self.market.N+1)
        self.cost = 0
        self.plot_path = plot_path
        prices = []
        for i in range(self.market.N):
            if self.day_tariff(i):
                price = self.market.day_price
            else:
                price = self.market.night_price
            prices.append([price])
        self.market.prices = np.array(prices)
        df = pd.read_csv(
            data_path + "ninja_pv_51.7520_-1.2578_uncorrected.csv", skiprows=3)

        start_date = self.initial_time - \
            timedelta(days=365*(self.initial_time.year - 2019))
        end_date = self.final_time - \
            timedelta(days=365*(self.final_time.year - 2019))

        df['date'] = df['time'].apply(
            lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M"))
        df1 = df[df['date'] >= start_date]
        df2 = df1[df1['date'] <= end_date]
        self.T_outside = df2['temperature'].values
        self.power_solar = df2['electricity'].values
        self.power = self.power_e + self.power_t + self.power_h

    def get_index(self, time: datetime) -> bool:
        """Find the closest index from datetime.

        Args:
            time: A datetime object of the simulation.

        Returns:
            An intenger between 0 and N and that encodes the time.

        Raises:
            Exception:time must be between initial and departure time.
        """
        if self.initial_time <= time <= self.date_end \
                + timedelta(hours=self.departure_time):
            duration = ((time - self.initial_time) / self.market.dt)
            return int(duration.total_seconds() // 3600)
        else:
            raise Exception("time must be between initial and departure time")

    def get_time(self, i: int) -> datetime:
        """Return time given an index

        Args:
            i: An integer between 0 and N.

        Returns:
            A datetime object that encodes the index.
        """
        return self.initial_time + timedelta(hours=(i * self.market.dt))

    def day_tariff(self, i: int) -> bool:
        """Returns a boolean refering to the electricity price of Economy 7.

        Args:
            i: An integer between 0 and N.

        Returns:
            A boolean that is true if an index refers to a valid time."""
        current_time = self.get_time(i)
        if current_time.hour < 24 and current_time.hour > 7:
            return True
        else:
            return False

    @timer
    def optimization(self) -> None:
        """Define the model and find the best consumption scenario."""
        cost = 0
        problem = pc.Problem()
        for i in range(self.market.N):
            # we can only charge the vehicule if it is in the house
            current_time = self.get_time(i)
            if self.departure_time <= current_time.hour < self.arrival_time:
                problem += self.power_e[i] == 0
            # power and energy must be positive
            else:
                problem += self.power_e[i] >= self.min_power
            problem += self.power_t[i] >= 0
            # evolution rule for temperature and energy storage
            if (i % 24 != 0 or i < 23) and i % 24 != self.arrival_time \
                    - self.departure_time - 1:
                problem += self.energy[i+1] == self.power_e[i] * \
                    self.market.dt + self.energy[i]
            elif i % 24 == self.arrival_time - self.departure_time - 1 and i < self.market.N:
                problem += self.energy[i+1] == self.E0[i//24]
            elif i % 24 == 0 and i >= 23:
                problem += self.energy[i] == self.EV_energy_goal
                problem += self.energy[i+1] == 0
            problem += self.temperature[i+1] == self.temperature[i] + self.market.dt * (
                self.beta * self.power_t[i] + self.alpha *
                (self.T_outside[i] - self.temperature[i]))
            # temperature must be between Tmin and Tmax
            problem += self.temperature[i+1] <= self.Tmax, \
                self.temperature[i+1] >= self.Tmin
            # power must be between min_power and max_power
            problem += self.power_e[i] <= self.max_power
            # adding the energy used cost
            cost += (self.power_t[i] + self.power_e[i] +
                     self.power_h[i] - self.power_solar[i]) * self.market.prices[i]
        problem += self.temperature[-1] == self.T0, \
            self.temperature[0] == self.T0
        problem += self.energy[-1] == self.EV_energy_goal
        problem.minimize = cost
        problem.solve(solver="cvxopt")
        self.cost = problem.value
        return self.cost

    @ timer
    def plot(self):
        """Display a graphical result of the optimization."""
        fig, ax = plt.subplots(2, 2)
        fig.set_size_inches((20, 5))

        hours = np.arange(self.initial_time, self.final_time + timedelta(hours=1),
                          np.timedelta64(1, 'h'))

        user_home_time = []
        t = [0, 0]
        for i in range(self.market.N):
            current_time = self.get_time(i)
            if self.departure_time <= current_time.hour < self.arrival_time:
                t = [i, i]
            else:
                t[0] = np.min([t[0], i])
                t[1] = np.max([t[1], i])
            if t not in user_home_time and t[0] != t[1]:
                user_home_time.append(t)

        # Plot price evolution
        ax[0, 0].plot(self.market.prices/100)
        ax[0, 0].set_xticks([])
        ax[0, 0].set_title(r"$Price$" + ' ' r"$£/kWh$", fontsize=15)

        # Plot power consumption
        ax[0, 1].plot(self.power_e.value, label='EV')
        ax[0, 1].plot(self.power_t.value, label='heating')
        ax[0, 1].plot(self.power_solar[:-1], label='solar')
        ax[0, 1].plot(self.power_h, label='other')
        ax[0, 1].set_xticks([])
        ax[0, 1].set_title(r"$Power$" + ' ' r"$kW$", fontsize=15)

        # Plot EV energy
        energy = self.energy.value
        for i in range(self.market.N):
            if i % 24 < self.arrival_time - self.departure_time:
                energy[i] = self.EV_energy_goal + (self.E0[i // 24] - self.EV_energy_goal) * (i % 24) / (
                    self.arrival_time - self.departure_time)
        ax[1, 0].plot(hours, 100 * energy/self.EV_energy_goal)
        ax[1, 0].set_yticks(ticks=np.linspace(0, 100, 3))
        ax[1, 0].set_title(r"$EV$" + ' ' + r"$battery$" +
                           ' ' + r"$\%$", fontsize=15)

        # Plot temperature
        ax[1, 1].plot(hours, self.temperature.value, label="House", color='g')
        ax[1, 1].plot(hours, self.T_outside, label="Outside", color='m')
        ax[1, 1].legend(bbox_to_anchor=(1, 1), prop={'size': 20})
        ax[1, 1].set_yticks(ticks=np.round(np.linspace(
            np.min(self.T_outside), self.Tmax + 3, 5), 0))
        ax[1, 1].set_title(r"$Temperature$" + ' ' r"$°C$", fontsize=15)

        # add shaded areas
        for i, t in enumerate(user_home_time):
            if i == len(user_home_time) - 1:
                ax[0, 1].axvspan(t[0], t[1], color='y', alpha=0.5,
                                 lw=0, label='user home')
            else:
                ax[0, 1].axvspan(t[0], t[1], color='y', alpha=0.5,
                                 lw=0)
        ax[0, 1].legend(bbox_to_anchor=(1, 1), prop={'size': 15})

        ax[0, 0].annotate(
            'a)',
            xy=(0, 1), xycoords='axes fraction',
            xytext=(+0.5, -0.5), textcoords='offset fontsize',
            fontsize=15, verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
        ax[0, 1].annotate(
            'b)',
            xy=(0, 1), xycoords='axes fraction',
            xytext=(+0.5, -0.5), textcoords='offset fontsize',
            fontsize=15, verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
        ax[1, 0].annotate(
            'c)',
            xy=(0, 1), xycoords='axes fraction',
            xytext=(+0.5, -0.5), textcoords='offset fontsize',
            fontsize=15, verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
        ax[1, 1].annotate(
            'd)',
            xy=(0, 1), xycoords='axes fraction',
            xytext=(+0.5, -0.5), textcoords='offset fontsize',
            fontsize=15, verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))

        plt.savefig(self.plot_path, bbox_inches='tight', format='pdf')

    @ timer
    def plot_vertical(self):
        """Display a graphical result of the optimization."""
        fig, ax = plt.subplots(4)
        fig.set_size_inches((8, 7))
        fig.tight_layout()

        hours = np.arange(self.initial_time, self.final_time + timedelta(hours=1),
                          np.timedelta64(1, 'h'))

        user_home_time = []
        t = [0, 0]
        for i in range(self.market.N):
            current_time = self.get_time(i)
            if self.departure_time <= current_time.hour < self.arrival_time:
                t = [i, i]
            else:
                t[0] = np.min([t[0], i])
                t[1] = np.max([t[1], i])
            if t not in user_home_time and t[0] != t[1]:
                user_home_time.append(t)

        # Plot price evolution
        ax[0].plot(self.market.prices/100)
        ax[0].set_xticks([])
        ax[0].set_title(r"$Price$" + ' ' r"$£/kWh$", fontsize=15)

        # Plot power consumption
        ax[1].plot(self.power_e.value, label='EV')
        ax[1].plot(self.power_t.value, label='heating')
        ax[1].plot(self.power_solar[:-1], label='solar')
        ax[1].plot(self.power_h, label='other')
        ax[1].set_xticks([])
        ax[1].set_title(r"$Power$" + ' ' r"$kW$", fontsize=15)

        # Plot EV energy
        energy = self.energy.value
        for i in range(self.market.N):
            if i % 24 < self.arrival_time - self.departure_time:
                energy[i] = self.EV_energy_goal + (self.E0[i // 24] - self.EV_energy_goal) * (i % 24) / (
                    self.arrival_time - self.departure_time)
        ax[2].plot(hours, 100 * energy/self.EV_energy_goal)
        ax[2].set_xticks([])
        ax[2].set_yticks(ticks=np.linspace(0, 100, 3))
        ax[2].set_title(r"$EV$" + ' ' + r"$battery$" +
                        ' ' + r"$\%$", fontsize=15)

        # Plot temperature
        ax[3].plot(hours, self.temperature.value, label="House", color='g')
        ax[3].plot(hours, self.T_outside, label="Outside", color='m')
        ax[3].legend(bbox_to_anchor=(1, 1), prop={'size': 10})
        ax[3].set_yticks(ticks=np.round(np.linspace(
            np.min(self.T_outside), self.Tmax + 3, 5), 0))
        ax[3].set_title(r"$Temperature$" + ' ' r"$°C$", fontsize=15)

        # add shaded areas
        for i, t in enumerate(user_home_time):
            if i == len(user_home_time) - 1:
                ax[1].axvspan(t[0], t[1], color='y', alpha=0.5,
                              lw=0, label='user home')
            else:
                ax[1].axvspan(t[0], t[1], color='y', alpha=0.5,
                              lw=0)
        ax[1].legend(bbox_to_anchor=(1, 1), prop={'size': 10})

        ax[0].annotate(
            'a)',
            xy=(0, 1), xycoords='axes fraction',
            xytext=(+0.5, -0.5), textcoords='offset fontsize',
            fontsize=15, verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
        ax[1].annotate(
            'b)',
            xy=(0, 1), xycoords='axes fraction',
            xytext=(+0.5, -0.5), textcoords='offset fontsize',
            fontsize=15, verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
        ax[2].annotate(
            'c)',
            xy=(0, 1), xycoords='axes fraction',
            xytext=(+0.5, -0.5), textcoords='offset fontsize',
            fontsize=15, verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))
        ax[3].annotate(
            'd)',
            xy=(0, 1), xycoords='axes fraction',
            xytext=(+0.5, -0.5), textcoords='offset fontsize',
            fontsize=15, verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1', edgecolor='none', pad=3.0))

        plt.savefig(self.plot_path,
                    bbox_inches='tight', format='pdf')

    @ timer
    def plot_separately(self):
        """Display a graphical result of the optimization."""
        hours = np.arange(self.initial_time, self.final_time + timedelta(hours=1),
                          np.timedelta64(1, 'h'))
        time = [str(pd.Timestamp(h).hour) for h in hours]
        user_home_time = []
        t = [0, 0]
        for i in range(self.market.N):
            current_time = self.get_time(i)
            if self.departure_time <= current_time.hour < self.arrival_time:
                t = [i, i]
            else:
                t[0] = np.min([t[0], i])
                t[1] = np.max([t[1], i])
            if t not in user_home_time and t[0] != t[1]:
                user_home_time.append(t)

        # Plot price evolution
        fig1, ax1 = plt.subplots()
        fig1.set_size_inches((15, 5))
        ax1.plot(self.market.prices/100)
        ax1.set_xticks([])
        ax1.set_ylabel(r"$Price$" + ' ' r"$£/kWh$", fontsize=25)
        ax1.tick_params(axis='both', labelsize=20)

        # Plot power consumption
        fig2, ax2 = plt.subplots()
        fig2.set_size_inches((15, 5))
        ax2.plot(self.power_e.value, '-x', label='EV')
        ax2.plot(self.power_t.value, '-s', label='heating')
        ax2.plot(self.power_solar[:-1], '--', label='solar')
        ax2.plot(self.power_h, '-o', label='other')
        ax2.tick_params(axis='both', labelsize=20)
        ax2.set_xticks([])
        ax2.set_ylabel(r"$Power$" + ' ' r"$kW$", fontsize=25)

        # Plot EV energy
        fig3, ax3 = plt.subplots()
        fig3.set_size_inches((15, 5))
        energy = self.energy.value
        for i in range(self.market.N):
            if i % 24 < self.arrival_time - self.departure_time:
                energy[i] = self.EV_energy_goal + (self.E0[i // 24] - self.EV_energy_goal) * (i % 24) / (
                    self.arrival_time - self.departure_time)
        ax3.plot(hours, 100 * energy/self.EV_energy_goal)
        ax3.tick_params(axis='both', labelsize=20)
        # ax3.set_xticklabels(time)
        ax3.set_yticks(ticks=np.linspace(0, 100, 3))
        ax3.set_ylabel(r"$EV$" + ' ' + r"$battery$" +
                       ' ' + r"$\%$", fontsize=25)
        ax3.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax3.xaxis.get_major_locator(), show_offset=False))
        ax3.set_xlabel(r"Time", fontsize=25)

        # Plot temperature
        fig4, ax4 = plt.subplots()
        fig4.set_size_inches((15, 5))
        ax4.plot(hours, self.temperature.value, '--', label="House", color='g')
        ax4.plot(hours, self.T_outside, '-x', label="Outside", color='m')
        ax4.tick_params(axis='both', labelsize=20)
        ax4.legend(bbox_to_anchor=(1, 1), prop={'size': 25})
        ax4.set_yticks(ticks=np.round(np.linspace(
            np.min(self.T_outside), self.Tmax + 3, 5), 0))
        # ax4.set_xticklabels(time)
        ax4.set_ylabel(r"$Temperature$" + ' ' r"$°C$", fontsize=25)
        ax4.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax4.xaxis.get_major_locator(), show_offset=False))
        ax4.set_xlabel(r"Time", fontsize=25)

        # add shaded areas
        for i, t in enumerate(user_home_time):
            if i == len(user_home_time) - 1:
                ax2.axvspan(t[0], t[1], color='y', alpha=0.5,
                            lw=0, label='user home')
            else:
                ax2.axvspan(t[0], t[1], color='y', alpha=0.5,
                            lw=0)
        ax2.legend(bbox_to_anchor=(1, 1), prop={'size': 25})

        fig1.savefig(self.plot_path.replace('.pdf', '1.pdf'),
                     bbox_inches='tight', format='pdf')
        fig2.savefig(self.plot_path.replace('.pdf', '2.pdf'),
                     bbox_inches='tight', format='pdf')
        fig3.savefig(self.plot_path.replace('.pdf', '3.pdf'),
                     bbox_inches='tight', format='pdf')
        fig4.savefig(self.plot_path.replace('.pdf', '4.pdf'),
                     bbox_inches='tight', format='pdf')


def tests(model: Model) -> None:
    """Test function of the class Model.

    Raises:
        Exception: get_index({current_time}) != {index})
    """
    N = model.market.N
    for i in range(N):
        current_time = model.get_time(i)
        j = model.get_index(current_time)
        if i != j:
            raise Exception(f"get_index({current_time}) = {j} != {i}")
