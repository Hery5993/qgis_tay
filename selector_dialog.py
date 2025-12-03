import inspect
import json
import os
import sys
from pathlib import Path
from typing import List, Dict

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import laspy
import numpy as np
from PyQt5.QtGui import *
from qgis._core import QgsMapLayer, QgsPointCloudLayer, QgsPointCloudStatistics, QgsRectangle
from qgis._gui import QgisInterface
from qgis.utils import iface

JSON_CFG = {
    "Intensité": {
        "type": QVariant.Int,
        "dimension": "Intensity",
        "laspy_dimension": "intensity",
        "widgets": {
            "RGB": False,
            "Min&Max": True,
            "Axe&Pas": True,
            "Symétrique": True,
            "Asymétrique": True,
            "Cycle": True
        }
    },
    "Altitude": {
        "type": QVariant.Double,
        "dimension": "Z",
        "laspy_dimension": "z",
        "widgets": {
            "RGB": False,
            "Min&Max": True,
            "Axe&Pas": True,
            "Symétrique": True,
            "Asymétrique": True,
            "Cycle": True
        }
    },
    "Couleurs": {
        "type": QVariant.Int,
        "dimension": "Red",
        "laspy_dimension": "red",
        "widgets": {
            "RGB": True,
            "Min&Max": True,
            "Axe&Pas": False,
            "Symétrique": False,
            "Asymétrique": False,
            "Cycle": False
        }
    },
    "Classification": {
        "type": QVariant.Int,
        "dimension": "Classification",
        "laspy_dimension": "classification",
        "widgets": {
            "RGB": False,
            "Min&Max": False,
            "Axe&Pas": False,
            "Symétrique": False,
            "Asymétrique": False,
            "Cycle": False
        }
    },
    "PointSourceId": {
        "type": QVariant.Int,
        "dimension": "PointSourceId",
        "laspy_dimension": "point_source_id",
        "widgets": {
            "RGB": False,
            "Min&Max": False,
            "Axe&Pas": False,
            "Symétrique": False,
            "Asymétrique": False,
            "Cycle": False
        }
    },
    "Scanner": {
        "type": QVariant.Int,
        "dimension": "ScannerChannel",
        "laspy_dimension": "scanner_channel",
        "widgets": {
            "RGB": False,
            "Min&Max": False,
            "Axe&Pas": False,
            "Symétrique": False,
            "Asymétrique": False,
            "Cycle": False
        }
    },
    "Normal Z": {
        "type": QVariant.Double,
        "dimension": "NormalZ",
        "widgets": {
            "RGB": False,
            "Min&Max": True,
            "Axe&Pas": True,
            "Symétrique": True,
            "Asymétrique": True,
            "Cycle": True
        }
    }
}


class _SpinBox(QWidget):

    def __init__(self, label: str, parent=None):
        QWidget.__init__(self, parent=parent)
        self._label = QLabel(label)

        # Mettre le calque
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.setLayout(lay)
        lay.addWidget(self._label)

        # Ajouter le widget
        self._integer = QSpinBox(self)
        self._integer.setRange(-10000000, +10000000)

        self._floating = QDoubleSpinBox(self)
        self._floating.setDecimals(4)
        self._floating.setRange(-10000000.0, +10000000.0)

        # Ajouter les widgets
        for it in [self._integer, self._floating]:
            lay.addWidget(it)
            it.setHidden(it == self._floating)

    @property
    def decimals(self):
        return self._floating.decimals()

    @decimals.setter
    def decimals(self, prec: int):
        self._floating.setDecimals(prec)

    @property
    def spin_height(self):
        return self._integer.height()

    @spin_height.setter
    def spin_height(self, value: int):
        self._integer.setFixedHeight(value)
        self._floating.setFixedHeight(value)

    @property
    def label(self):
        return self._label.text()

    @label.setter
    def label(self, val: str):
        if val:
            self._label.setText(val)

    def to_integer(self, label: str = None):
        """
        Afficher les widgets entiers
        Args:
            label (list): Labels des inputs
        """
        self._integer.setVisible(True)
        self._floating.setHidden(True)
        self.label = label

    def to_float(self, label: str = None):
        """
        Afficher les widgets double
        Args:
            label (list): Liste des labels pour les floats
        """
        self._floating.setVisible(True)
        self._integer.setHidden(True)
        self.label = label

    @property
    def value(self):
        """
        Méthode pour retourner la valeur du widget
        Returns:
            Valeur du widget (Entier ou Floating)
        """
        for it in [self._integer, self._floating]:
            if it.isVisible():
                return it.value()
        return None

    def is_integer(self):
        """
        Méthode pour déterminer si le spinbox est un entier
        Returns:
            True si le spin box est un entier
        """
        return self._integer.isVisible()

    def is_floating(self):
        """
        Méthode pour déterminer si le spinbox est un float
        Returns:
            True si le spin box est un float
        """
        return self._floating.isVisible()

    @value.setter
    def value(self, val):
        if self._integer.isVisible():
            self._integer.setValue(int(val))
        else:
            self._floating.setValue(float(val))


class _SpinBoxN(QWidget):

    def __init__(self, labels: dict, parent=None):
        QWidget.__init__(self, parent)
        self._nb = len(labels)

        # Créer le widget
        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        self._widgets = list()
        for label in labels:
            widget = _SpinBox(label, parent)
            self._widgets.append(widget)
            self.layout().addWidget(widget)

        self.labels = labels

    @property
    def spin_height(self):
        if len(self._widgets):
            return self._widgets[0].spin_height()
        return None

    @property
    def widgets(self) -> List[_SpinBox]:
        return self._widgets

    @spin_height.setter
    def spin_height(self, value: int):
        for spin in self._widgets:
            spin.spin_height = value

    @property
    def labels(self):
        return [it.label for it in self._widgets]

    @labels.setter
    def labels(self, labels):
        if not labels or not isinstance(labels, (list, dict)) or not len(labels):
            return

        try:
            if isinstance(labels, list):
                for index_val, label in enumerate(labels):
                    try:
                        self._widgets[index_val].label = label
                    except:
                        continue

            elif isinstance(labels, dict):
                for index_val, label in enumerate(labels.keys()):
                    try:
                        self._widgets[index_val].label = label
                        value = labels[label]
                        if isinstance(value, int):
                            if value == QVariant.Int:
                                self._widgets[index_val].to_integer()
                            elif value == QVariant.Double:
                                self._widgets[index_val].to_float()
                    except:
                        continue

        except Exception as e:
            return

    def _convert(self, func, index_val: int = None, label=None):
        if not self._widgets:
            return

        def launch_conv(widg: _SpinBox):
            """
            Convertir le type de données traitée par le widget
            """
            try:
                spin_meth = getattr(widg, func)
                if spin_meth:
                    spin_meth(label)
            except Exception:
                pass

        # Convertir tous les widgets
        if index_val is None or not isinstance(index_val, int):
            for it in self._widgets:
                launch_conv(it)

        # Convertir juste certain widget
        else:
            if index_val >= len(self._widgets):
                index_val = len(self._widgets) - 1

            # Lancer le converter
            launch_conv(self._widgets[index_val])

    def to_integer(self, index_val: int = None, label: str = None):
        self._convert("to_integer", index_val=index_val, label=label)

    def to_float(self, index_val: int = None, label: str = None):
        self._convert("to_float", index_val=index_val, label=label)

    def values(self):
        """
        Fonction permettant de récupérer les valeurs des widgets
        """
        try:
            return {it.label: it.value for it in self._widgets}
        except Exception:
            return {}

    def set_default(self, *args):
        for index_val, value in enumerate(args):
            try:
                self._widgets[index_val].value = value
            except Exception as e:
                pass

    def set_values(self, values: dict):
        if not isinstance(values, dict):
            return

        for key, value in values.items():
            try:
                idx = self.labels.index(key)
                self._widgets[idx].value = value
            except Exception:
                continue

    def spinbox_size(self):
        return len(self._widgets)


class ParametersWidget(QWidget):

    def __init__(self, parent=None):
        QWidget.__init__(self, parent=parent)

        # Information de debugage
        self.debug = True
        # self.attrs = [it["laspy_dimension"] for _, it in JSON_CFG.items()]

        # Liste des dimensions
        lay = QHBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        self.setLayout(lay)

        self._dimensions = QComboBox(self)
        self._dimensions.setFixedHeight(27)
        self.layout().addWidget(self._dimensions)

        # Liste des méthodes
        self._methods = QComboBox(self)
        self._methods.setFixedHeight(27)
        self.layout().addWidget(self._methods)

        # Créer les widgets
        self._widgets = {
            "RGB": _SpinBoxN(
                labels={
                    "RMin": QVariant.Int,
                    "RMax": QVariant.Int,
                    "GMin": QVariant.Int,
                    "GMax": QVariant.Int,
                    "BMin": QVariant.Int,
                    "BMax": QVariant.Int
                },
                parent=self
            ),
            "Min&Max": _SpinBoxN(
                labels={
                    "Min": QVariant.Int,
                    "Max": QVariant.Int
                },
                parent=self
            ),
            "Axe&Pas": _SpinBoxN(
                labels={
                    "Axe": QVariant.Int,
                    "Pas": QVariant.Int
                },
                parent=self
            ),
            "Symétrique": _SpinBoxN(
                labels={
                    "Axe": QVariant.Int,
                    "Facteur": QVariant.Int
                },
                parent=self
            ),
            "Asymétrique": _SpinBoxN(
                labels={
                    "Axe": QVariant.Double,
                    "Pas-": QVariant.Double,
                    "Pas+": QVariant.Double
                },
                parent=self
            ),
            "Cycle": _SpinBoxN(
                labels={
                    "Pas": QVariant.Double
                },
                parent=self
            )
        }

        # Ajouter les widgets au layout
        for method_, widget in self._widgets.items():
            self.layout().addWidget(widget)
            widget.setVisible(False)
            widget.spin_height = 27

        # Connecter les signaux pour afficher les méthodes et les widgets
        self._dimensions.currentIndexChanged.connect(lambda: self.show_method())
        self._methods.currentIndexChanged.connect(lambda: self.show_widget())

    def pdal_dimension(self):
        if not self._dimensions.count():
            return None

        return self._dimensions.currentData()["dimension"]

    def laspy_dimension(self):
        if not self._dimensions.count():
            return None

        return self._dimensions.currentData()["laspy_dimension"]

    @property
    def method(self):
        return self._methods.currentText()

    @property
    def dimensions(self):
        return [self._dimensions.itemText(idx) for idx in range(self._dimensions.count())]

    @dimensions.setter
    def dimensions(self, values):
        if not isinstance(values, list):
            return

        values = [it for it in values if isinstance(it, str) and it in JSON_CFG.keys()]
        self._dimensions.clear()

        for it in values:
            self._dimensions.addItem(it, JSON_CFG[it])

        if self._dimensions.count() > 0:
            self.show_method()

    def show_method(self):
        try:
            # Récupérer les données
            list_data_ = self._dimensions.currentData()
            if not list_data_:
                return

            list_method_ = [it for it, active in list_data_["widgets"].items() if active]

            # Afficher les données
            self._methods.blockSignals(True)  #
            self._methods.clear()
            self._methods.addItems(list_method_)
            self._methods.blockSignals(False)

            # Afficher le widgets correspondant à la méthode courante
            self.show_widget()
        except Exception as e:
            if self.debug:
                print(inspect.currentframe().f_code.co_name, " ==> err: ", e)

    def show_widget(self):
        try:
            current_method = self._methods.currentText()
            data_type = self._dimensions.currentData()["type"]

            # Cacher tous les widgets
            for widget in self._widgets.values():
                widget.setVisible(False)

            # Afficher le widget correspondan
            if current_method in self._widgets:
                if data_type == QVariant.Int:
                    self._widgets[current_method].to_integer()
                else:
                    self._widgets[current_method].to_float()

                self._widgets[current_method].setVisible(True)
        except Exception as e:
            if self.debug:
                print(inspect.currentframe().f_code.co_name, " ==> err: ", e)

    @property
    def value(self):
        if not self._methods.count():
            return dict()

        return self._widgets[self._methods.currentText()].values()

    @value.setter
    def value(self, val):
        try:
            for meth, widge in self._widgets.items():
                for k, v in val.items():
                    if k in widge.labels:
                        widge.set_values({k: v})
        except Exception:
            pass


class CurrentLayer(QObject):
    """
    calculer la statistique de la couche courante sur le projet
    """
    g_stat = dict()
    z_stat = dict()
    attrs = [it["laspy_dimension"] for it in JSON_CFG.values() if "laspy_dimension" in it.keys()]
    trans = {it["dimension"]: key for key, it in JSON_CFG.items()}
    dimensions = list()

    layerChanged = pyqtSignal(list)

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        self._layer: QgsMapLayer = iface.activeLayer()
        iface.currentLayerChanged.connect(lambda layer: self.set_layer(layer))

    @property
    def layer(self):
        return self._layer

    @property
    def global_stat(self):
        if not self._layer:
            return None

        layer_path = self._layer.dataProvider().dataSourceUri()
        if layer_path not in CurrentLayer.g_stat.keys():
            return None

        return CurrentLayer.g_stat[layer_path]

    @property
    def zonal_stat(self):
        if not self._layer:
            return None

        layer_path = self._layer.dataProvider().dataSourceUri()
        if layer_path not in CurrentLayer.z_stat.keys():
            return None

        return CurrentLayer.z_stat[layer_path]

    def _vpc_stats(self, files: List[Dict]):
        """
        Compute aggregated stats (min/max) for multiple COPC files.
        """
        stat_details = dict()
        # Parcourir les fichiers et récupérer les statistiques
        for file in files:
            filepath = os.path.join(file['directory'], file['filename'])
            if not os.path.exists(filepath):
                continue

            layer = QgsPointCloudLayer(filepath, "copc_layer", "copc")
            if not layer.isValid():
                continue

            # Récupérer la statistique de la couche
            layer_stat = self.stat_json(layer)
            if not layer_stat:
                continue

            stat_details[filepath] = layer_stat
        return stat_details

    def set_layer(self, layer: QgsMapLayer):
        if not layer:
            return

        layer_path = layer.dataProvider().dataSourceUri()
        if layer_path not in CurrentLayer.g_stat.keys():
            if layer_path.endswith(".vpc"):
                details_st = dict()
                with open(layer_path, "r") as _vpc:
                    json_obj = json.load(_vpc)

                if "stats" in json_obj.keys():
                    details_st = json_obj["stats"]
                else:
                    entries = self.parse_layer(layer_path)
                    globals_ = self._vpc_stats(entries)

                    for file, dim_values in globals_.items():
                        try:
                            for dim, values in dim_values.items():
                                if dim not in details_st.keys():
                                    details_st[dim] = {
                                        "minimum": [values["minimum"]],
                                        "maximum": [values["maximum"]],
                                        "mean": [values["mean"]]
                                    }
                                else:
                                    details_st[dim]["minimum"].append(values["minimum"])
                                    details_st[dim]["maximum"].append(values["maximum"])
                                    details_st[dim]["mean"].append(values["mean"])
                        except Exception:
                            continue

                    for dim, vals in details_st.items():
                        details_st[dim]["minimum"] = min(details_st[dim]["minimum"])
                        details_st[dim]["maximum"] = max(details_st[dim]["maximum"])
                        details_st[dim]["mean"] = np.median(details_st[dim]["mean"])

                    # Ecrire dans le fichier
                    json_obj["stats"] = details_st

                    # Enregistrer le fichier
                    with open(layer_path, "w", encoding="utf-8") as f:
                        json.dump(json_obj, f, indent=4)
            else:
                details_st = self.stat_json(layer)

            self._layer = layer
            CurrentLayer.g_stat[layer_path] = details_st.copy()

        # Récupérer les dimensions valides
        try:
            pdals = list()
            for dimension, values in CurrentLayer.g_stat[layer_path].items():
                if values["minimum"] == values["maximum"]:
                    continue
                else:
                    pdals.append(dimension)

            dimes = [v for k, v in CurrentLayer.trans.items() if k in pdals]
            if CurrentLayer.dimensions != dimes:
                CurrentLayer.dimensions = dimes
                self.layerChanged.emit(CurrentLayer.dimensions)
        except Exception:
            pass

    def parse_layer(self, source: str):
        """
        Parse STAC/GeoJSON file and extract each .copc.laz asset with its bbox.
        Returns a list of dicts with 'filename' and 'bbox'.
        """
        try:
            if source.endswith(".vpc") or source.endswith(".json"):
                copc_files = []
                dire = os.path.dirname(source)
                geojson = dict()
                with open(source, "r") as _reader:
                    geojson = json.load(_reader)

                for feature in geojson.get("features", []):
                    asset = feature.get("assets", {}).get("data", {})
                    href = asset.get("href", "")
                    bbox = feature.get("properties", {}).get("proj:bbox", {})
                    if href.endswith(".copc.laz"):
                        copc_files.append({
                            "filename": (Path(dire) / Path(href)).resolve(strict=False).__str__(),
                            "bbox": QgsRectangle(bbox[0], bbox[1], bbox[3], bbox[4]),
                            "zrange": (bbox[2], bbox[5]),
                            "directory": os.path.dirname(source)
                        })

                # Ne retourner que les fichiers contenus dans le boundiung box courant du projet
                return copc_files

            # Traiter les fichiers autres que vpc
            elif Path(source).suffix in [".laz", ".las"]:
                copc_files = []
                with laspy.open(source) as stream_:
                    header = stream_.header
                    copc_files.append({
                        "filename": source,
                        "bbox": QgsRectangle(header.x_min, header.y_min, header.x_max, header.y_max),
                        "zrange": (header.z_min, header.z_max),
                        "object": None,
                        "directory": os.path.dirname(source)
                    })

                return copc_files
            else:
                return []
        except Exception as e:
            return []

    def _files_in_box(self, box: QgsRectangle, copc_entries):
        """
        Given a bbox, return files that intersect or contain it.
        Updates global min/max extent (_mins/_maxs).
        """
        cnt_files = []
        for entry in copc_entries:
            if entry["bbox"].contains(box) or entry["bbox"].intersects(box):
                cnt_files.append(entry)
        return cnt_files

    def stat_json(self, layer: QgsPointCloudLayer):
        """
        Méthode pour récupérer la statistique d'une couche sous forme d'un json
        Args:
            layer (QgsPointCloudLayer): Couche avec laquelle on va calculer la statistique
        Returns:
            Statistique de la couche
        """
        """ Fonction pour prendre les statistics de la couche"""
        if not layer or not isinstance(layer, QgsPointCloudLayer):
            return None

        # Ne pas continuer tant que le calcul des stats de la couche n'est pas totalement terminé
        while layer.statisticsCalculationState() != 2:
            QCoreApplication.processEvents()

        stat: QgsPointCloudStatistics = layer.statistics()
        stat_json: QByteArray = stat.toStatisticsJson()

        # Transformer en dict
        try:
            datas: dict = json.loads(stat_json.data())
            return datas['stats']
        except Exception as e:
            return None


class Selector(ParametersWidget):

    def __init__(self, parent=None):
        # Paramètrer les widget
        ParametersWidget.__init__(self, parent)
        self.layer = CurrentLayer()
        self.layer.layerChanged.connect(lambda dimes: self.set_dimensions(dimes))

        # Connecter les signaux
        self._dimensions.currentTextChanged.connect(self._set_default)
        self._methods.currentTextChanged.connect(self._set_default)

    def set_dimensions(self, dimes):
        self.dimensions = dimes

    def set_default(self):
        if not self.layer.layer:
            return
