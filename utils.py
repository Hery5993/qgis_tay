import abc
import contextlib
import json
import math
import os.path
import queue
import random
import sys
import threading
import time

from functools import wraps
import datetime
import hashlib
import os
import subprocess
import uuid
import re

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict
from typing import List

import laspy
import matplotlib.pyplot as plt
import numpy as np
from qgis.PyQt.QtCore import QVariant, QRegExp, QSize, QRect
from qgis.PyQt.QtGui import QRegExpValidator, QDoubleValidator, QIntValidator, QLinearGradient, QBrush, QColor, QPen
from qgis.PyQt.QtWidgets import QSpinBox, QDoubleSpinBox, QComboBox, QFileDialog, QStyle, QStyledItemDelegate, \
    QColorDialog
from qgis.PyQt.QtCore import QObject, Qt, pyqtSignal, QRunnable, QThreadPool, QTimer
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QHBoxLayout, QVBoxLayout, QMessageBox, QToolButton, QAction, QMenu, QDialog, \
    QFormLayout, \
    QPushButton, QWidget, QListWidget, QListWidgetItem, QProgressBar, QLineEdit, QApplication
from qgis.core import Qgis, QgsProject, QgsApplication, QgsPointCloudLayer, QgsLayerTree, QgsLayerTreeGroup, \
    QgsRectangle
from qgis.core import QgsField, QgsFields, QgsVectorLayer, QgsFeature, QgsGeometry, QgsVectorFileWriter, QgsWkbTypes
from qgis.core import QgsGradientColorRamp, QgsGradientStop, QgsColorRamp, QgsLimitedRandomColorRamp, \
    QgsColorBrewerColorRamp, QgsPresetSchemeColorRamp, QgsStyle
from qgis.gui import QgsColorRampButton
from qgis.gui import QgsFileWidget
from qgis.utils import iface
from qgis.PyQt.QtWidgets import QMessageBox
from qgis.PyQt.QtCore import Qt


ICON_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ressources", "icons")
UI_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ressources", "ui")

_MD5 = "c0d99a0d8e05d86cf4073500ca9d7124"
_EXE_PATH = r"C:\Futurmap\Outils\EXE\Undermap\validation"
_EXE_NAME = "verify_licence.exe"


class VerifyLicence:

    def __init__(self):
        self.exe_file = None

    def verify(self):
        """
        Vérifie la validité de la licence.
        """
        self.exe_file = self.get_exe_file()

        # Vérifier si l'exe file existe
        if not self.exe_file:
            return 0

        # Verifier s'ils ont le même checksum
        if not self.verify_checksum():
            return 0

        return self.is_licence_valid()

    @staticmethod
    def is_licence_valid():
        """
        Vérifie si la licence est valide en exécutant le fichier exécutable.
        """
        exe = os.path.join(_EXE_PATH, _EXE_NAME)
        if not os.path.exists(exe):
            return 0

        try:
            # Vérifier si la licence est valide
            try:
                result = subprocess.Popen(
                    [exe, "NUAGE_QGIS"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    shell=True,
                    encoding="cp1252"
                )
            except Exception as e:
                result = subprocess.Popen(
                    [exe, "NUAGE_QGIS"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    shell=True
                )

            stdout, _ = result.communicate()  # Bloque jusqu'à la fin du
            if "'count': 0," in stdout:
                return 0

            return 1

        except Exception as e:
            print("Except :", e)
        return 0

    @staticmethod
    def pc_mac():
        """
        Récupère l'adresse MAC de la machine.
        """
        mac_int = uuid.getnode()
        mac_hex = '{:012x}'.format(mac_int)

        mac_adresse = re.findall('..', mac_hex)
        adresse_mac = ':'.join(mac_adresse)

        return adresse_mac

    def verify_checksum(self):
        """
        Vérifie si le checksum du fichier exécutable correspond au checksum valide.
        :return: True si valide, False sinon.
        """
        return str(self.checksum(self.exe_file)).strip() == _MD5

    @staticmethod
    def get_md5_in_file(file):
        """
        Obtenir le checksum MD5 d'un fichier.
        :param file: fichier à analyser.
        :return: checksum MD5.
        """
        with open(file, 'r', encoding="UTF-8") as f:
            content = f.readlines()
            for line in content:
                if "md5" in line.lower():
                    md5 = line.split('=')[1].strip()
                    return md5.replace('"', "")

    @staticmethod
    def checksum(file):
        """
        Calcule le checksum MD5 du fichier exécutable.
        :return:
        """
        return hashlib.md5(open(file, 'rb').read()).hexdigest()

    @staticmethod
    def get_exe_file():
        """
            Parcourir le répertoire de l'exécutable pour trouver un fichier.
        """

        if not os.path.exists(_EXE_PATH):
            return ""

        for root, dirs, files in os.walk(_EXE_PATH):
            for f in files:
                if f == _EXE_NAME:
                    return os.path.join(root, f)


def verify_licence(func):
    """
        Créer un décorateur pour vérifier la licence avant d'exécuter une fonction.
        func : fonction à décorer.
    """
    @wraps(func)
    def is_valid(*args, **kwargs):
        verifier = VerifyLicence()
        res = verifier.verify()

        # Si valide
        if res and isinstance(res, int) and res == 1:
            return func(*args, **kwargs)
        return None

    return is_valid


def zero_margin_layout(horizontal: bool = True):
    """
    Retourner une nouvelle layout sans marge
    Args:
        horizontal(bool): Booléen permettant de connaitre le type de layout à créer
    Returns:
        le Layout sans margins
    """
    lay = QHBoxLayout() if horizontal else QVBoxLayout()
    lay.setContentsMargins(0, 0, 0, 0)
    return lay


class CloudUtils:
    """
    Classe permettant de rassembler toutes les fonctions utilitaires
    """

    @staticmethod
    def gis_info(msg: str):
        """
        Pusher une information dans la barre de message
        Args:
            msg (str): Message à pusher
        """
        iface.messageBar().pushMessage(msg, Qgis.MessageLevel.Info, 0)

    @staticmethod
    def gis_warning(msg: str):
        """
        Pusher un warning dans la barre de message
        Args:
            msg (str): Message à pusher
        """
        iface.messageBar().pushMessage(msg, Qgis.MessageLevel.Warning, 0)

    @staticmethod
    def gis_error(msg):
        """
        Pusher une erreur dans la barre de message
        Args:
            msg (str): Sujet de l'erreur à pusher
        """
        iface.messageBar().pushMessage(msg, Qgis.MessageLevel.Critical, 0)

    @staticmethod
    def project_path():
        """
        Fonction permettant de rétourner le chemin du projet
        Returns:
            Chemin du projet (str)
        """
        return QgsProject.instance().readPath("/")

    @staticmethod
    def project_file_path():
        """
        Fonction permettant de rétourner le chemin complet du projet
        Returns:
            Le chemin complet menant au projet sur le disque
        """
        return QgsProject.instance().fileName()

    @staticmethod
    def project_is_valid():
        """
        Vérifier si le projet est valide
        """
        return CloudUtils.project_path() not in ["./", "/"]

    @staticmethod
    def save_project():
        """
        Fonction permettant de sauvegarder le projet QGIS
        Returns:
            True si le projet est sauvegardé avec succès
        """
        # Vérifier si le projet est valide
        if not CloudUtils.project_is_valid():
            return False

        # Ecrire le Projet
        project = QgsProject.instance()
        project.write(CloudUtils.project_file_path())
        return True

    @staticmethod
    def push_info(value: str, title: str = "Information"):
        """
        Fonction pour logger une info dans QGIS
        Args:
            value (str): L'info à afficher
            title (str): Titre de la dialogue
        """
        QMessageBox.information(
            iface.mapCanvas(),
            title,
            value
        )

    @staticmethod
    def pdal_wrench_exec_path():
        """
        Fonction permettant de récupérer le chemin complet vers pdal_wrench
        Returns:
            Chemin complet de l'executable pdal_wrench
        """
        return QgsApplication.libexecPath() + "pdal_wrench.exe"

    @staticmethod
    def pdal_exec_path():
        return QgsApplication.libexecPath() + "pdal.exe"

    @staticmethod
    def push_warning(value: str, title: str = "Avertissement"):
        """
        Fonction pour logger un avertissement dans QGIS
        Args:
            value (str): L'avertissement à afficher
            title (str): Titre de la dialogue
        """
        QMessageBox.warning(
            iface.mapCanvas(),
            title,
            value
        )

    @staticmethod
    def push_error(value: str, title: str = "Erreur"):
        """
        Fonction pour logger une erreur dans QGIS
        Args:
            value (str): L'erreur à afficher
            title (str): Titre de la dialogue
        """
        QMessageBox.critical(
            iface.mapCanvas(),
            title,
            value
        )

    @staticmethod
    def icon_path(icon_name: str):
        """
        Fonction permettant de récupérer une icone en particulier
        Args:
            icon_name (str): Nom de l'image servant comme icone

        Returns:
            str Le chemin vers l'icone
        """
        return os.path.join(ICON_PATH, icon_name)

    @staticmethod
    def append_point_cloud(cloud: QgsPointCloudLayer):
        """
        Fonction permettant d'ajouter toujours un nuages au dessous des tous les vecteurs
        dans un Projet QGIS
        Args:
            cloud (QgsPointCloudLayer): Le nuage à ajouter
        """
        # Si le point cloud n'est pas valide (rétourner)
        if not isinstance(cloud, QgsPointCloudLayer):
            return

        # Ajouter le nuage de points en dessous de tous les vecteurs contenu dans le projet QGIS
        # Récupérer l'index max d'un vecteur dans un projet
        project: QgsProject = QgsProject.instance()
        root: QgsLayerTree = project.layerTreeRoot()
        for item in root.children():
            pass

    @staticmethod
    def sort_layers_in_group(parent_group):
        """
        Fonction permettant de trier les couches dans un groupe
        Args:
            parent_group (str | QgsLayerTreeGroup): Groupe à trier

        Returns:

        """
        # Récupérer le groupe principale
        root_group: QgsLayerTree = QgsProject.instance().layerTreeRoot()
        if isinstance(parent_group, str):
            group = root_group.findGroup(parent_group)
        elif isinstance(parent_group, QgsLayerTreeGroup):
            group = parent_group
        else:
            return

        # Trier les couches dans le groupe
        # Filtrer l'affichage des réseaux dans le groupe RSX
        layer_list = [c for c in group.children()]
        layer_list = sorted(layer_list, key=lambda x: x.name(), reverse=True)

        # Rajouter les couches
        for child in layer_list:
            clone = child.clone()
            group.insertChildNode(0, clone)
            group.removeChildNode(child)

    @staticmethod
    def sort_groups_in_group(parent_group):
        """
        Fonction permettant de trier les groupes dans un groupe parent
        Args:
            parent_group (str | QgsLayerTreeGroup): Groupe à trier

        Returns:

        """
        # Récupérer le groupe principale
        root_group: QgsLayerTree = QgsProject.instance().layerTreeRoot()
        if isinstance(parent_group, str):
            group = root_group.findGroup(parent_group)
        elif isinstance(parent_group, QgsLayerTreeGroup):
            group = parent_group
        else:
            return

        # Filtrer l'affichage des réseaux dans le groupe TIF
        group_list = [c for c in group.children()]
        group_list = sorted(group_list, key=lambda x: x.name(), reverse=True)

        for child in group_list:
            clone = child.clone()
            group.insertChildNode(0, clone)
            group.removeChildNode(child)

    @staticmethod
    def convert_rgb(r1, g1, b1, fitColor=True):
        """
        Fonction permettant de fit les valeurs des paramètres rgb
        Args:
            r1 (int):
            g1 (int):
            b1 (int):
            fitColor (bool):

        Returns:

        """
        if fitColor:
            r2 = math.ceil(r1 / 65536 * 256)
            g2 = math.ceil(g1 / 65536 * 256)
            b2 = math.ceil(b1 / 65536 * 256)
        else:
            r2, g2, b2 = r1, g1, b1
        return r2, g2, b2

    @staticmethod
    def random_color():
        """
        Générer une couleur aléatoire au format HEXA
        Returns:
            La coleur hex en str
        """
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return f"#{r:02X}{g:02X}{b:02X}"

    @staticmethod
    def confirm(title: str, msg: str):
        # Confirmer par oui ou non
        _box = QMessageBox()
        _box.setWindowTitle(title)
        _box.setText(msg)
        _box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        _box.setDefaultButton(QMessageBox.No)
        ret = _box.exec()

        if ret == QMessageBox.Yes:
            return True
        else:
            return False

    @staticmethod
    def parse_copc_features(geojson: Dict, source: str) -> List[Dict]:
        """
        Parse STAC/GeoJSON file and extract each .copc.laz asset with its bbox.
        Returns a list of dicts with 'filename' and 'bbox'.
        """
        try:
            if source.endswith(".vpc") or source.endswith(".json"):
                copc_files = []
                dire = os.path.dirname(source)

                for feature in geojson.get("features", []):
                    asset = feature.get("assets", {}).get("data", {})
                    href = asset.get("href", "")
                    bbox = feature.get("properties", {}).get("proj:bbox", {})
                    if href.endswith(".copc.laz"):
                        copc_files.append({
                            "filename": (Path(dire) / Path(href)).resolve(strict=False),
                            "bbox": QgsRectangle(bbox[0], bbox[1], bbox[3], bbox[4]),
                            "zrange": (bbox[2], bbox[5]),
                            "object": None,
                            "directory": os.path.dirname(source)
                        })
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

    @staticmethod
    def request_files_in_bbox(box: QgsRectangle, copc_entries: List[Dict], valid: bool = False) -> Dict:
        """
        Given a bbox, return files that intersect or contain it.
        Updates global min/max extent (_mins/_maxs).
        """
        if not valid:
            return dict()

        _bins = None
        _mins, _maxs = None, None
        _hist = None

        cnt_files = []
        for entry in copc_entries:
            if entry["bbox"].contains(box) or entry["bbox"].intersects(box):
                cnt_files.append(entry)

                if _mins is None:
                    _mins = [
                        entry["bbox"].xMinimum(),
                        entry["bbox"].yMinimum(),
                        entry["zrange"][0]
                    ]
                    _maxs = [
                        entry["bbox"].xMaximum(),
                        entry["bbox"].yMaximum(),
                        entry["zrange"][1]
                    ]
                else:
                    _mins = [
                        min(entry["bbox"].xMinimum(), _mins[0]),
                        min(entry["bbox"].yMinimum(), _mins[1]),
                        min(entry["zrange"][0], _mins[2])
                    ]
                    _maxs = [
                        max(entry["bbox"].xMaximum(), _maxs[0]),
                        max(entry["bbox"].yMaximum(), _maxs[1]),
                        max(entry["zrange"][1], _maxs[2])
                    ]

        val = {
            "files": cnt_files,
            "bins": _bins,
            "mins": _mins,
            "maxs": _maxs,
            "hist": _hist
        }

        return val


@contextlib.contextmanager
def change_directory(path: str):
    """
    Changer d'un dossier
    Args:
        path (str): Chemin du dossier à se rendre
    """
    d = os.getcwd()
    os.chdir(path)

    try:
        yield
    finally:
        os.chdir(d)


class DefaultActionHandler(QObject):
    """
    Classe permettant de manipuler les actions par défaut
    """

    def __init__(self, p=None):
        QObject.__init__(self, p)
        self._tools = set()

    def add_tool(self, btn: QToolButton):
        """
        Ajouter une action correspondante à un tool bouton
        Args:
            btn (QToolButton): L'outil à activer
        """
        self._tools.add(btn)

    def watch(self):
        """
        Fonction permettant de connecter les actions aux tool button
        """
        # Connecter les signals des actions
        for tool in self._tools:
            menu: QMenu = tool.menu()
            for act in menu.actions():
                act.triggered.connect(lambda _, a=act, b=tool: self.set_default(a, b))

    def set_default(self, action: QAction, tool: QToolButton):
        """
        Fonction permettant de définir une action comme action par défaut d'un 'tool button'
        Args:
            action (QAction): L'action à définir
            tool (QToolButton): Le 'tool button', menu déroulante
        """
        tool.setDefaultAction(action)


class SelectFile(QDialog):
    """
    Classe permettant de sélectionner des fichiers ou dossier
    """

    def __init__(
            self,
            title: str = "Selectionner",
            label: str = "Sélectionner les fichiers",
            filtre: str = "All *.*",
            initial_dir: str = "",
            p=None
    ):
        # Initialiser la dialogue
        QDialog.__init__(self, p)
        self._label = label
        self._filter = filtre
        self._initial_dir = initial_dir
        self.setWindowTitle(title)
        self.setMinimumWidth(400)

        # Créer le layout pour sélectionner les fichiers ou le dossier
        layout = QVBoxLayout()
        form = QFormLayout()
        self._file_widget = QgsFileWidget()
        self._file_widget.setStorageMode(QgsFileWidget.StorageMode.GetMultipleFiles)
        self._file_widget.setFilter(self._filter)
        form.addRow(self._label, self._file_widget)

        # Former les boutons de validations
        btnl = QHBoxLayout()
        btnl.addStretch()

        validate_btn = QPushButton(self.tr("Valider"))
        cancel_btn = QPushButton(self.tr("Annuler"))
        btnl.addWidget(validate_btn)
        btnl.addWidget(cancel_btn)

        # Mettre à jour le layout de la dialogue
        layout.addLayout(form)
        layout.addLayout(btnl)
        self.setLayout(layout)

        # Connecter les boutons
        validate_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)

    def setMode(self, mode: QgsFileWidget.StorageMode):
        """
        Changer le mode de sélection des fichiers ou dossier
        Args:
            mode (QgsFileWidget.StorageMode): Mode de sélection des fichiers ou dossier
        """
        self._file_widget.setStorageMode(mode)

    def selected_files(self):
        """
        Récupérer la liste des fichiers sélectionnés par l'utilisateur
        Returns:
            list, la liste des fichiers sélectionnés par l'utilisateur
        """
        return self._file_widget.splitFilePaths(self._file_widget.filePath())


class ListWidget(QWidget):

    def __init__(self, list_values: list, p=None, list_max_height: int = 100):
        QWidget.__init__(self, p)
        self._list_max_height = list_max_height
        self._list_values = list_values

        # Ajouter les widgets
        box = QHBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        self.list_widget = QListWidget()
        self.list_widget.setMaximumHeight(self._list_max_height)
        box.addWidget(self.list_widget)

        # Ajouter les options de droite
        right = QVBoxLayout()
        self.select_all = QPushButton(self.tr("Sélectionner tout"))
        self.invert_selection = QPushButton(self.tr("Inverser la sélection"))
        self.deselect_all = QPushButton(self.tr("Désélectionner tout"))
        right.addWidget(self.select_all)
        right.addWidget(self.invert_selection)
        right.addWidget(self.deselect_all)
        right.addStretch()
        box.addLayout(right)

        # Setter le layout principal
        self.setLayout(box)

        # Populate list
        self._populate()

        """Connecte les signaux des boutons."""
        self.select_all.clicked.connect(lambda : self._select_all())
        self.deselect_all.clicked.connect(lambda : self._deselect_all())
        self.invert_selection.clicked.connect(lambda : self._invert_selection())


    def _select_all(self):
        """Coche tous les éléments."""
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setCheckState(Qt.Checked)

    def _deselect_all(self):
        """Décoche tous les éléments."""
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setCheckState(Qt.Unchecked)

    def _invert_selection(self):
        """Inverse l'état de chaque élément."""
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                item.setCheckState(Qt.Unchecked)
            else:
                item.setCheckState(Qt.Checked)

    def selected_items(self):
        """
        Fonction permettant de récupérer la listes des entités séléctionnés
        """
        selected_ = list()
        row_count = self.list_widget.count()
        index_ = 0
        while row_count > 0:
            item: QListWidgetItem = self.list_widget.item(index_)
            if item.checkState() == Qt.Checked:
                selected_.append(item.text())

            index_, row_count = index_ + 1, row_count - 1
        return selected_

    def _populate(self):
        """
        Fonction permettant d'hydrater une liste widget
        """
        # Ajoutez les noms des couches du type de géométrie spécifié à la liste avec des cases à cocher
        for value in self._list_values:
            item = QListWidgetItem(value)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setToolTip(value)
            self.list_widget.addItem(item)


class DialogValidation(QWidget):

    def __init__(self, p=None, add_progressbar: bool = False, accept_msg: str = "Valider", reject_msg: str = "Annuler"):
        # Initialiser le widget
        QWidget.__init__(self, p)
        self._add_progress = add_progressbar
        self._accept_msg = accept_msg
        self._reject_msg = reject_msg
        self.progress_bar = QProgressBar()

        # Ajouter les widgets
        box = QHBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        if self._add_progress:
            box.addWidget(self.progress_bar)

        box.addStretch()
        btn_accept = QPushButton(self.tr(accept_msg))
        btn_reject = QPushButton(self.tr(reject_msg))
        box.addWidget(btn_accept)
        box.addWidget(btn_reject)
        self.setLayout(box)

        # Connecter les signals
        if self.parent() and isinstance(self.parent(), QDialog):
            btn_accept.clicked.connect(self.parent().accept)
            btn_reject.clicked.connect(self.parent().reject)


class WorkerSignals(QObject):
    """Signaux communs pour tous les workers"""
    finished = pyqtSignal(object, str)  # résultat, message
    started = pyqtSignal(object, str)
    calulated = pyqtSignal(object, str)
    progress = pyqtSignal(int, int)  # current, total
    error = pyqtSignal(str)  # message d'erreur


class BaseWorker(QRunnable):
    """Classe de base abstraite pour tous les workers"""

    def __init__(self, file: dict, bounds: list, dimension: str = "z"):
        QRunnable.__init__(self)
        self.signals = WorkerSignals()

        # Données communes
        self._queue = queue.Queue()
        self._abort = False
        self._file = file
        self._bounds = bounds
        self._dimension = dimension
        self._lock = threading.Lock()

        # Résultats (à définir dans les sous-classes)
        self._result = None

        # Statistiques
        self._processed_count = 0
        self._error_count = 0
        self._start_time = None

        # Remplir la queue
        for bound in self._bounds:
            self._queue.put((self._file, bound))

        self._stat = {
            "classification": [sys.float_info.max, -sys.float_info.max],
            "z": [sys.float_info.max, -sys.float_info.max],
            "intensity": [sys.float_info.max, -sys.float_info.max],
            "scanner_channel": [sys.float_info.max, -sys.float_info.max],
            "point_source_id": [sys.float_info.max, -sys.float_info.max],
            "red": [sys.float_info.max, -sys.float_info.max],
            "green": [sys.float_info.max, -sys.float_info.max],
            "blue": [sys.float_info.max, -sys.float_info.max],
        }

        # Lister les médianes des chaque dimensions
        self._medianes = {
            key: [] for key, _ in self._stat.items()
        }

    @property
    def finished(self):
        return self.signals.finished

    @property
    def progress(self):
        return self.signals.progress

    @property
    def error(self):
        return self.signals.error

    def cancel(self):
        """Annuler le worker"""
        self._abort = True

    def is_canceled(self):
        """Vérifier si annulé"""
        return self._abort

    @property
    def dim(self):
        return self._dimension

    @dim.setter
    def dim(self, val: str):
        self._dimension = val

    @contextlib.contextmanager
    def query_context(self, entry: dict, bound):
        query_result = None
        try:
            with laspy.CopcReader.open(entry["filename"]) as copc:
                query_result = copc.query(bound)

                # Remplir les informations relatives au nuage
                if len(query_result):
                    for it in [dim for dim in self._stat.keys()]:
                        try:
                            temp_ = getattr(query_result, it)
                            self._stat[it][0] = min(self._stat[it][0], np.min(temp_))
                            self._stat[it][1] = max(self._stat[it][1], np.max(temp_))
                            if it in self._medianes.keys():
                                self._medianes[it].append(np.median(temp_))
                        except Exception as e:
                            continue

                yield query_result
        except Exception as e:
            self.signals.error.emit(f"Erreur worker: (*1) {e}")
        finally:
            # Libérer la mémoire
            if query_result is not None:
                del query_result

    @abc.abstractmethod
    def process_query(self, query_result, file_info: dict):
        """
        Méthode abstraite pour traiter le résultat d'une requête
        Args:
            query_result: Résultat de copc.query()
            file_info: Informations sur le fichier
        """
        pass

    @abc.abstractmethod
    def get_result(self):
        """Retourner le résultat final"""
        pass

    def run(self):
        """Méthode principale d'exécution"""
        self._start_time = time.time()
        total_tasks = self._queue.qsize()

        try:
            while not self._queue.empty() and not self.is_canceled():
                try:
                    file, bound = self._queue.get_nowait()
                    if self.is_canceled():
                        continue

                    try:
                        # Utiliser le context manager pour la requête
                        with self.query_context(file, bound) as query_result:
                            self.process_query(query_result, file)

                        self._processed_count += 1
                    except Exception as e:
                        self.error.emit(f"Erreur worker: (*2) {e}")
                        continue

                    # Émettre le signal de progression
                    current = self._processed_count + self._error_count
                    self.signals.progress.emit(current, total_tasks)
                except Exception as e:
                    self.error.emit(f"Erreur worker: (*3) {e}")
                    self._error_count += 1

            # Calcul du temps d'exécution
            execution_time = time.time() - self._start_time if self._start_time else 0

            # Message final
            if self.is_canceled():
                msg = f"Annulé '{self._file['filename']}'"
            else:
                msg = f"Terminé '{self._file['filename']}' [{execution_time:.2f}s] - {self._processed_count} succès, {self._error_count} erreurs"

        except Exception as e:
            execution_time = time.time() - self._start_time if self._start_time else 0
            msg = f"Erreur critique '{self._file['filename']}' [{execution_time:.2f}s]: {str(e)}"
            self.error.emit(f"Erreur worker: (*4) {msg}")
        finally:
            # Toujours émettre le signal final
            result = self.get_result()
            self.signals.finished.emit(result, msg)


class HistogramWorker(BaseWorker):
    """Worker pour calculer un histogramme"""

    def __init__(
            self, file: dict, bounds: list,
            bins, dimension: str = "z"
    ):
        super().__init__(file, bounds, dimension)
        self._bins = bins
        self._hist = None
        self._histograms = dict()

        try:
            for key, _ in self._stat.keys():
                self._histograms[key] = None
        except Exception as e:
            self.signals.error.emit(f"Erruersdfl: {e}")

    def process_query(self, query_result, file_info: dict):
        """Traiter la requête pour l'histogramme"""
        try:
            data = getattr(query_result, self._dimension)
        except Exception as e:
            self.signals.error.emit(f"Erreur worker: (*5) {e}")
            return

        # Calculer l'histogramme
        tmp_hist, _ = np.histogram(data, bins=self._bins[self._dimension])

        # Mettre à jour l'histogramme total (thread-safe)
        with self._lock:
            for dimension, bin in self._bins.items():
                dt = getattr(query_result, dimension)
                hist, _ = np.histogram(dt, bins=bin)
                if self._histograms[dimension] is None:
                    self._histograms[dimension] = hist
                else:
                    self._histograms[dimension] = self._histograms[dimension] + hist

            if self._hist is None:
                self._hist = tmp_hist
            else:
                self._hist += tmp_hist

    def get_result(self):
        """Retourner l'histogramme final"""
        with self._lock:
            return {
                'histogram': self._hist.copy() if self._hist is not None else None,
                'bins': self._bins.copy() if self._bins is not None else None,
                'dimension': self._dimension,
                'total_points': np.sum(self._hist) if self._hist is not None else 0,
                "histograms": self._histograms
            }


class StatisticsWorker(BaseWorker):
    """Worker pour calculer des statistiques descriptives"""

    def __init__(self, file: dict, bounds: list, dimension: str = "z"):
        super().__init__(file, bounds, dimension)

        self._values = []
        self._count = 0
        self._min_val = float('inf')
        self._max_val = float('-inf')

    def process_query(self, query_result, file_info: dict):
        """Traiter la requête pour les statistiques"""
        # Récupérer les données selon la dimension
        try:
            # Récupérer les données
            data = getattr(query_result, self._dimension)
        except Exception as e:
            self.signals.error.emit(f"Erreur worker: (*7) {e}")
            return None

        # Calculer les statistiques incrementales (thread-safe)
        with self._lock:
            self._count += len(data)
            self._min_val = min(self._min_val, np.min(data))
            self._max_val = max(self._max_val, np.max(data))

            # Récupérer les statistiques supplémentaires
            self._stat[self._dimension][0] = min(self._stat[self._dimension][0], np.min(data))
            self._stat[self._dimension][1] = max(self._stat[self._dimension][1], np.max(data))

    @property
    def statistics(self):
        return self._stat

    @property
    def medianes(self):
        return self._medianes

    def get_result(self):
        """Retourner les statistiques finales"""
        with self._lock:
            if self._count == 0:
                return None

            return {
                'dimension': self._dimension,
                'count': self._count,
                'min': self._min_val,
                'max': self._max_val,
                'range': self._max_val - self._min_val,
                "global": self.statistics,
                "medianes": self._medianes
            }


class PointCloudParse(QObject):
    """
    Class to parse a VPC file and manage COPC files for querying.
    """
    statQueryFinished = pyqtSignal()
    statQueryStarted = pyqtSignal()
    histQueryFinished = pyqtSignal()
    histQueryStarted = pyqtSignal()
    queryError = pyqtSignal(object)

    def __init__(self, file: str):
        QObject.__init__(self, parent=None)

        # Variables pour lancer les workers
        self._workers = list()
        self._runnings = list()

        # Variables de al
        self._file = file
        self._req_files = None
        self._req_finish = []
        self._valid = os.path.exists(self._file) and Path(self._file).suffix in [".json", ".vpc", ".laz", ".las"]
        self._lock = threading.Lock()
        try:
            # Charger depuis un fichier
            if Path(self._file).suffix in [".vpc", ".json"]:
                with open(self._file, "r") as f:
                    data = json.load(f)
            else:
                data = None

            # Récupérer la liste des fichiers à traiter
            self._copc_entries = CloudUtils.parse_copc_features(data, self._file)

        except Exception as e:
            self._valid = False
            self._copc_entries = []
            self.queryError.emit(f"Erreur: (#1) {e}")

        # Récupérer le référence du canvas
        canvas = iface.mapCanvas()

        self._nb_bounds = 5
        self._bounds = []

        # Paramètres de l'histogramme
        self._bins_nb = 1000
        self._bins = None
        self._hist = None

        # Min/Max XYZ
        self._maxs = None
        self._mins = None
        self._stat = {
            "classification": [sys.float_info.max, -sys.float_info.max],
            "z": [sys.float_info.max, -sys.float_info.max],
            "intensity": [sys.float_info.max, -sys.float_info.max],
            "scanner_channel": [sys.float_info.max, -sys.float_info.max],
            "point_source_id": [sys.float_info.max, -sys.float_info.max],
            "red": [sys.float_info.max, -sys.float_info.max],
            "green": [sys.float_info.max, -sys.float_info.max],
            "blue": [sys.float_info.max, -sys.float_info.max],
        }

        # Créer l'histogramme
        self._all_histograms = {
            key: None for key in self._stat.keys()
        }

        # Déclarer les médianes
        self._medianes = {
            key: [] for key in self._stat.keys()
        }

        # Pool de threads
        self.pool = QThreadPool()

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, rect: QgsRectangle):
        self._bounds = []
        if not rect or not isinstance(rect, QgsRectangle):
            return

        # Calculer intervalles
        x_edges = np.linspace(rect.xMinimum(), rect.xMaximum(), self._nb_bounds + 1)
        y_edges = np.linspace(rect.yMinimum(), rect.yMaximum(), self._nb_bounds + 1)

        # Créer les découpages
        for i in range(self._nb_bounds):
            for j in range(self._nb_bounds):
                self._bounds.append(
                    laspy.Bounds(
                        mins=(x_edges[i], y_edges[j]),
                        maxs=(x_edges[i + 1], y_edges[j + 1])
                    )
                )

    @property
    def copc_entries(self) -> List[Dict]:
        """
        Getter for COPC entries.
        """
        return self._copc_entries

    def is_valid(self) -> bool:
        return self._valid

    def histogram(self):
        return self._hist

    def _need_compute(self, selector: dict):
        if not selector:
            return True

        # Vérifier que les dimensions et la méthode a besoin de calcul statistique

        # Si selector comprend l'option auto retourner True

    def _launch_compute(self, dimension: str = "z", stat: bool = False):
        """
        Méthode pour lancer la détermination du renderer
        Args:
            dimension: Dimension à traiter
            stat: si ne veux que le compute de la statistique
        Returns:
            None
        """
        # Computer les bins d'abord
        self._workers = list()
        self._runnings = list()
        if stat:
            self.statQueryStarted.emit()
        else:
            self.histQueryStarted.emit()

        # Lancer les processus
        for file in self._req_files:
            # Créer le worker
            if stat:
                worker = StatisticsWorker(
                    file, self.bounds,
                    dimension=dimension
                )
            else:
                worker = HistogramWorker(
                    file, self.bounds,
                    dimension=dimension,
                    bins={
                        dim: self.bins(dim) for dim in self._stat.keys()
                    }
                )

            # Connecter les workers
            worker.setAutoDelete(True)
            worker.error.connect(lambda msg: self.queryError.emit(msg))
            worker.finished.connect(self._update_bins if stat else self._update_hist)

            # Collecter le worker
            self._workers.append(worker)

        # Lancer les bins
        for worker in self._workers:
            self.pool.start(worker)

    @property
    def statistics(self):
        return self._stat

    @property
    def medianes(self):
        return self._medianes

    def bins(self, dimension: str):
        if (
                dimension in self._stat.keys()
                and self._stat[dimension][0] != sys.float_info.max
                and self._stat[dimension][1] != -sys.float_info.max
        ):
            return np.linspace(
                self._stat[dimension][0],
                self._stat[dimension][1],
                self._bins_nb + 1
            )
        return None

    def _update_bins(self, result, msg):
        """
        Mettre à jour la statistique
        """
        with self._lock:
            # Mettre à jour les statistiques globales
            for dim, value in self._stat.items():
                value[0] = min(result["global"][dim][0], value[0])
                value[1] = max(result["global"][dim][1], value[1])

            # Afficher la dialogue
            self._runnings.append(result)

            # Mettre à jour les bins
            self._bins = self.bins(result["dimension"])

            # Mettre à jour les statistiques globales
            for dim, value in result["medianes"].items():
                self._medianes[dim].extend(value)

            try:
                for dim, hist in result["global"]["histograms"].items():
                    if self._all_histograms[dim] is None:
                        self._all_histograms[dim] = hist
                    else:
                        self._all_histograms[dim] += hist
            except:
                pass

        if len(self._runnings) == len(self._workers):
            self.statQueryFinished.emit()

    def _launch_workers(self):
        for worker in self._workers:
            self.pool.start(worker)

    def _update_hist(self, result, msg: str):
        """
        Fonction permettant de mettre à jour l'histogramme
        Args:
            result: nouveau histogramme
            msg: Message à afficher sur la dialogue
        """
        # Afficher la dialogue
        with self._lock:
            self._runnings.append(result)
            try:
                # Mettre à jour l'histogramme
                if self._hist is None:
                    self._hist = result["histogram"]
                else:
                    self._hist += result["histogram"]
            except Exception as e:
                self.queryError.emit(f"Erreur: (#3) {e}")

            # Vérifier la condition d'arrêt
            if len(self._runnings) == len(self._workers):
                self.histQueryFinished.emit()

    def _create_histogram(self):
        """
        Crée un renderer optimisé basé sur l'histogramme et les bins

        Args:
            histogram: array numpy de l'histogramme
            bins: array numpy des bins
            attribute_name: nom de l'attribut à utiliser pour le rendu
        """
        # Validation
        if not self._all_histograms:
            return

        bins = self.bins("z")
        plt.hist(
            x=bins[:-1],  # Les centres ou le bord gauche (ici, on utilise le bord gauche pour l'alignement)
            bins=bins,  # Les bords des bins
            weights=self._all_histograms["z"],  # Les effectifs/fréquences pour chaque bin
            histtype='bar',  # Type de tracé : 'bar' pour un histogramme classique
            edgecolor='black'  # Ajoute une bordure noire pour mieux distinguer les bins
        )

        plt.title("Histogramme avec bins et poids pré-calculés")
        plt.xlabel("Valeurs de Z")
        plt.ylabel("Fréquence/Effectif")
        plt.show()

    def compute_renderer(
            self, box: QgsRectangle,
            dimension: str = "z",
            selector_values=None
    ):
        """
        Méthode pour lancer la détermination du renderer
        Args:
            box: Bounding box du canvas
            dimension: Dimension à traiter
            selector_values: Paramètres du lancement
        Returns:
            None
        """
        # Vérifier la validité
        if not self.is_valid():
            return

        # Calculer les découpages
        self.bounds = box

        # Récupérer les fichiers qui passent par le bounding box du canvas
        res: dict = CloudUtils.request_files_in_bbox(box, self._copc_entries, self.is_valid())
        self._req_files, self._bins, self._mins, self._maxs, self._hist = res["files"], res["bins"], res["mins"], res[
            "maxs"], res["hist"]

        # Computer d'abord le min max de la dimension
        self._launch_compute(dimension, True)

        # Lancer après
        def cb_func():
            self._launch_compute(dimension)
            self.statQueryFinished.disconnect()

        def cb_hist():
            self._create_histogram()
            self.histQueryFinished.disconnect()

        # Lancer le calcul de l'histogram
        self.statQueryFinished.connect(cb_func)
        self.histQueryFinished.connect(cb_hist)

    def compute_statistics(
            self, box: QgsRectangle,
            dimension: str = "z"
    ):
        """
        Méthode pour lancer la détermination du renderer
        Args:
            box: Bounding box du canvas
            dimension: Dimension à traiter
        Returns:
            None
        """
        # Vérifier la validité
        if not self.is_valid():
            return

        # Calculer les découpages
        self.bounds = box

        # Récupérer les fichiers qui passent par le bounding box du canvas
        res: dict = CloudUtils.request_files_in_bbox(
            box,
            self._copc_entries,
            self.is_valid()
        )

        self._req_files, self._bins, self._mins, self._maxs, self._hist = res["files"], res["bins"], res["mins"], res[
            "maxs"], res["hist"]

        # Computer d'abord le min max de la dimension
        self._launch_compute(dimension, True)


class DisplayAndCopy(QWidget):
    initial_css = """
        QPushButton{
            border: 1px solid gray;
            border-radius: 2px;
        }
    """

    clicked_css = """
        QPushButton{
            border: 1px solid gray;
            border-radius: 2px;
        }

        QPushButton{
            background: gray;
        }
    """

    def __init__(self, p, default_val: str = "default", margin: int = 0, justify=Qt.AlignRight):
        QWidget.__init__(self, p)
        layout = QHBoxLayout()
        self.setLayout(layout)
        self.setStyleSheet(self.initial_css)

        # Créer le label
        self.label = QLineEdit()
        self.label.setAlignment(justify)
        self.label.setReadOnly(True)

        layout.addWidget(self.label)
        self.label.setText(default_val)
        layout.setContentsMargins(margin, margin, margin, margin)

        # Ajouter le bouton copier
        self.btn = QPushButton(QIcon(":images/themes/default/mActionEditCopy.svg"), "")
        self.btn.setToolTip("Copier la valeur")
        self.btn.setMaximumWidth(20)
        layout.addWidget(self.btn)

        self.btn.setCheckable(True)
        self.btn.clicked.connect(self.to_clipboard)

        self.btn_time = QTimer()
        self.btn_time.setInterval(500)
        self.btn_time.timeout.connect(self.copy_finished)

    def setText(self, value: str):
        """
        Mettre à jour la valeur du label
        Args:
            value (str): Valeur à afficher
        Returns:
            None
        """
        self.label.setText(value)

    def to_clipboard(self):
        """
        Copier la valeur dans le presse papier
        1. Démarrer le timer
        2. Changer le style du bouton
        3. Copier la valeur dans le presse papier
        Returns:
            None
        """
        self.btn_time.start()
        self.setStyleSheet(self.clicked_css)
        clipboard = QApplication.clipboard()
        clipboard.setText(self.label.text())

    def copy_finished(self):
        """
        Slot appelé lorsque le timer est écoulé
        """
        self.setStyleSheet(self.initial_css)
        self.btn_time.stop()


class _Editor(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.line_edit = QLineEdit(self)
        self.btn_browse = QPushButton("...", self)
        self.btn_browse.setFixedWidth(25)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.line_edit)
        layout.addWidget(self.btn_browse)

    def text(self):
        return self.line_edit.text()

    def setText(self, text):
        try:
            self.line_edit.setText(text)
        except:
            pass

    def setGeometry(self, rect: QRect, *__args):
        self.line_edit.setFixedHeight(rect.height())
        self.line_edit.setFixedWidth(rect.width() - rect.height())
        self.btn_browse.setFixedHeight(rect.height())
        self.btn_browse.setFixedWidth(rect.height())
        super().setGeometry(rect, *__args)


class _BaseDelegate(QStyledItemDelegate):

    def __init__(self, _default_val, parent=None):
        QStyledItemDelegate.__init__(self, parent=parent)
        self._default_value = _default_val

    @property
    def default_value(self):
        return self._default_value

    @default_value.setter
    def default_value(self, value):
        self._default_value = value


class _ColorEditor(_Editor):

    def __init__(self, parent=None, current_color=None):
        _Editor.__init__(self, parent)
        self._current_color = current_color
        self.btn_browse.clicked.connect(self._choose_color)

    def _choose_color(self):
        # Ouvrir le dialogue de couleur
        color = QColorDialog.getColor(self._current_color, self.parent(), "Choisir une couleur")

        # Si une couleur valide a été choisie, la sauvegarder immédiatement
        if color.isValid():
            self.line_edit.setText(color.name().upper())
        else:
            try:
                self.line_edit.setText(self._current_color.name().upper())
            except Exception:
                self.line_edit.setText("#FFFFFF")


class ColorRampDelegate(QStyledItemDelegate):
    def __init__(self, default_ramp=None, parent=None):
        super().__init__(parent)
        # valeur par défaut : dégradé rouge → bleu
        self.default_ramp = default_ramp or QgsGradientColorRamp(QColor("red"), QColor("blue"))

    def createEditor(self, parent, option, index):
        btn = QgsColorRampButton(parent)
        btn.setShowGradientOnly(True)
        ramp = index.data(Qt.EditRole)
        if isinstance(ramp, QgsGradientColorRamp):
            btn.setColorRamp(ramp)
        else:
            btn.setColorRamp(self.default_ramp)
        btn.colorRampChanged.connect(lambda _: self.commitData.emit(btn))
        return btn

    def setEditorData(self, editor, index):
        ramp = index.data(Qt.EditRole)
        if isinstance(ramp, QgsGradientColorRamp):
            editor.setColorRamp(ramp)
        else:
            editor.setColorRamp(self.default_ramp)

    def setModelData(self, editor, model, index):
        ramp = editor.colorRamp()
        model.setData(index, ramp, Qt.EditRole)

    def paint(self, painter, option, index):
        value = index.data(Qt.EditRole) or self.default_ramp
        painter.save()

        if isinstance(value, QgsGradientColorRamp):
            stops = value.stops()
            rect = option.rect
            grad = QLinearGradient(rect.left(), rect.top(), rect.right(), rect.top())
            for grad_stop in stops:
                grad_stop: QgsGradientStop = grad_stop
                grad.setColorAt(grad_stop.offset, grad_stop.color)
            painter.fillRect(rect, QBrush(grad))
        painter.restore()

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)

    @property
    def default_value(self):
        return self.default_ramp

    @default_value.setter
    def default_value(self, value):
        self.default_ramp = value


class OptionsDelegate(_BaseDelegate):
    """
    CODE/002
    """
    """Délégué Boolean simplifié"""

    def __init__(
            self,
            options: list = None,
            default_index: int = 0,
            parent=None
    ):
        self._options = options if options and len(options) else ["True", "False"]
        self._default_index = default_index if 0 <= default_index < len(self._options) else 0
        try:
            opt = self._options[self._default_index]
        except Exception:
            opt = options[0]

        super().__init__(opt, parent)

    def createEditor(self, parent, option, index):
        combobox = QComboBox(parent)
        combobox.addItems(self._options)
        return combobox

    def setEditorData(self, editor, index):
        current_text = str(index.data(Qt.DisplayRole) or "True")
        combo_index = editor.findText(current_text)
        if combo_index >= 0:
            editor.setCurrentIndex(combo_index)
        else:
            editor.setCurrentIndex(self._default_index)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentText(), Qt.EditRole)


class NumberDelegate(_BaseDelegate):
    """
    CODE/002
    """
    """Délégué pour afficher un SpinBox (entier ou décimal)"""

    def __init__(self, default_value, minimum=0, maximum=100, decimals=2, parent=None, strict: bool = True):
        super().__init__(default_value, parent)
        self.is_integer = isinstance(default_value, int)
        self.minimum = minimum
        self.maximum = maximum
        self.decimals = decimals
        self.strict = strict

        self._suffix = ""
        self._prefix = ""

    @property
    def suffix(self):
        return self._suffix

    @property
    def prefix(self):
        return self._prefix

    @suffix.setter
    def suffix(self, val: str):
        self._suffix = val

    @prefix.setter
    def prefix(self, val: str):
        self._prefix = val

    def _set_validator(self, number: QLineEdit):
        if self.strict:
            if self.is_integer:
                validator = QIntValidator(self.minimum, self.maximum)
                number.setValidator(validator)
            else:
                validator = QDoubleValidator(float(self.minimum), float(self.maximum), self.decimals)
                number.setValidator(validator)
        else:
            validator = QRegExpValidator(QRegExp(r"^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$"))
            # validator = QRegExpValidator(QRegExp(r"^(%[^\W_]|[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?(%[^\W_])?)$"))
            number.setValidator(validator)

    def createEditor(self, parent, option, index):
        """Créer l'éditeur SpinBox approprié"""
        number = QLineEdit(parent)
        self._set_validator(number)
        return number

    def setEditorData(self, editor, index):
        """Charger la valeur actuelle dans l'éditeur"""
        try:
            value = index.data(Qt.DisplayRole)
            if value is not None:
                if self.strict:
                    if self.is_integer:
                        editor.setText(str(int(float(value))))
                    else:
                        editor.setText(str(float(value)))
                else:
                    editor.setText(value)
            else:
                editor.setText(str(self.minimum))
        except (ValueError, TypeError):
            editor.setText(str(self.minimum))

    def setModelData(self, editor, model, index):
        """Sauvegarder la valeur du SpinBox"""
        value = editor.text()
        model.setData(index, value, Qt.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        """Ajuster la géométrie de l'éditeur"""
        editor.setGeometry(option.rect)

    def paint(self, painter, option, index):
        """Personnaliser l'affichage selon le type de valeur"""
        value = index.data(Qt.DisplayRole)

        # Formater la valeur selon le type
        if value is not None:
            try:
                if self.strict:
                    if self.is_integer:
                        display_text = str(int(float(str(value))))
                    else:
                        display_text = f"{float(str(value)):.{self.decimals}f}"
                else:
                    display_text = value
            except (ValueError, TypeError):
                display_text = str(self.minimum)
        else:
            display_text = str(self.minimum)

        # Dessiner le fond
        if option.state & QStyle.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())
            text_color = option.palette.highlightedText().color()
        else:
            painter.fillRect(option.rect, option.palette.base())
            text_color = option.palette.text().color()

        # Dessiner le texte aligné à droite (style numérique)
        painter.setPen(text_color)
        font = painter.font()
        font.setFamily("Consolas, Monaco, monospace")  # Police monospace pour les nombres
        painter.setFont(font)

        text_rect = option.rect.adjusted(5, 0, -5, 0)  # Marges
        painter.drawText(text_rect, Qt.AlignRight | Qt.AlignVCenter, display_text)

    def sizeHint(self, option, index):
        """Définir la taille préférée"""
        return QSize(80, 25)


class FormattedNumberDelegate(_BaseDelegate):
    """
    CODE/002
    """

    def __init__(
            self, default_val,
            is_float=False,
            prefix="",
            suffix="",
            parent=None
    ):
        super().__init__(default_val, parent)
        self.is_integer = isinstance(default_val, int)
        self.is_float = is_float
        self.prefix = prefix
        self.suffix = suffix

    def createEditor(self, parent, option, index):
        if self.is_integer and self.is_float:
            return QLineEdit(parent)  # priorité à QLineEdit
        elif self.is_integer:
            editor = QSpinBox(parent)
            editor.setMinimum(-9999999999)
            editor.setMaximum(999999999)
            return editor
        elif self.is_float:
            editor = QDoubleSpinBox(parent)
            editor.setMinimum(-9999999999.0)
            editor.setMaximum(9999999999.0)
            editor.setDecimals(4)
            return editor
        else:
            return QLineEdit(parent)

    def setEditorData(self, editor, index):
        raw_value = index.model().data(index, Qt.EditRole)

        # Nettoyage de la valeur si jamais elle contient déjà suffix/prefix (par précaution)
        if isinstance(raw_value, str):
            value = raw_value.replace(self.prefix, "").replace(self.suffix, "").strip()
        else:
            value = raw_value

        if isinstance(editor, QSpinBox):
            editor.setValue(int(float(value)))
        elif isinstance(editor, QDoubleSpinBox):
            editor.setValue(float(value))
        elif isinstance(editor, QLineEdit):
            editor.setText(str(value))

    def setModelData(self, editor, model, index):
        if isinstance(editor, (QSpinBox, QDoubleSpinBox)):
            value = editor.value()
        elif isinstance(editor, QLineEdit):
            value = editor.text()
        else:
            value = editor.text()

        model.setData(index, value, Qt.EditRole)

    def displayText(self, value, locale):
        # Ajout du préfixe/suffixe uniquement à l'affichage
        try:
            number = float(value)
            # Formatage avec séparateur de milliers et 2 décimales
            formatted = f"{number:,.2f}"
            return f"{self.prefix}{formatted}{self.suffix}"
        except (ValueError, TypeError):
            return str(value)


class _FileEditor(_Editor):

    def __init__(self, parent=None, mode: str = "open", file_filter: str = "All *.*"):
        _Editor.__init__(self, parent)
        self.mode = mode
        self.file_filter = file_filter

        # Ouvrir ou sauvegarder unfichier
        self.btn_browse.clicked.connect(self.open_or_save_file)

    def open_or_save_file(self):
        """
        Fonction permettant d'ouvrir un fichier ou d'en sauvegarder un
        Returns:
            None
        """
        current_path = self.line_edit.text() or ""
        if self.mode == "open":
            file_path, _ = QFileDialog.getOpenFileName(self, "Open File", current_path, self.file_filter)
        elif self.mode == "save":
            file_path, _ = QFileDialog.getSaveFileName(self, "Save File", current_path, self.file_filter)
        else:
            file_path = ""

        if file_path:
            self.line_edit.setText(file_path)


class FileDelegate(_BaseDelegate):
    def __init__(self, default_file: str, mode="open", file_filter="All Files (*.*)", parent=None):
        super().__init__(default_file, parent)
        self.mode = mode
        self.file_filter = file_filter
        self.default_file = default_file

    def createEditor(self, parent, option, index):
        editor = _FileEditor(parent, mode=self.mode, file_filter=self.file_filter)
        editor.setText(self.default_file if self.default_file else "")
        editor.setGeometry(option.rect)
        return editor

    def setEditorData(self, editor, index):
        current_value = index.model().data(index, Qt.EditRole) or ""
        editor.setText(current_value)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.text(), Qt.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)


class ColorDelegate(QStyledItemDelegate):
    """Delegate permettant l'édition d'une couleur (hex string) via un champ + bouton qui ouvre QColorDialog."""

    def createEditor(self, parent, option, index):
        editor_widget = QWidget(parent)
        layout = QHBoxLayout(editor_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        line = QLineEdit(editor_widget)
        btn = QPushButton("...", editor_widget)
        btn.setMaximumWidth(28)
        layout.addWidget(line)
        layout.addWidget(btn)
        editor_widget.line = line
        editor_widget.btn = btn

        # connecter le bouton pour ouvrir le dialog
        def open_color_dialog():
            current = QColor(line.text()) if line.text() else QColor("#FFFFFF")
            color = QColorDialog.getColor(current, editor_widget, "Choisir une couleur")
            if color.isValid():
                line.setText(color.name())

        btn.clicked.connect(open_color_dialog)
        return editor_widget

    def setEditorData(self, editor, index):
        value = index.model().data(index, Qt.DisplayRole)
        editor.line.setText(value if value is not None else "")

    def setModelData(self, editor, model, index):
        text = editor.line.text()
        # ensure valid hex
        if not text:
            text = "#FFFFFF"
        color = QColor(text)
        if not color.isValid():
            # try to fallback to white
            text = "#FFFFFF"
        model.setData(index, text, Qt.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)


class GradientDelegate(QStyledItemDelegate):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.default_ramp = "Viridis"

    def createEditor(self, parent, option, index):
        btn = QgsColorRampButton(parent)
        btn.setShowGradientOnly(True)

        ramp = self.deserialize(index.data(Qt.EditRole))
        if not ramp:
            ramp = self.ramp_by_name(self.default_ramp)

        btn.setColorRamp(ramp)

        # Commit automatique quand la rampe change
        btn.colorRampChanged.connect(lambda: self.commitData.emit(btn))
        btn.colorRampChanged.connect(lambda: self.closeEditor.emit(btn))
        return btn

    def setEditorData(self, editor, index):
        ramp = self.deserialize(index.data(Qt.EditRole))
        editor.setColorRamp(ramp or self.ramp_by_name(self.default_ramp))

    def setModelData(self, editor, model, index):
        ramp = editor.colorRamp()
        serialized = self.serialize(ramp)
        json_data = json.dumps(serialized, indent=None)
        model.setData(index, json_data, Qt.EditRole)

    def paint(self, painter, option, index):
        value = index.data(Qt.EditRole)
        painter.save()
        rect = option.rect.adjusted(2, 2, -2, -2)

        # Tenter de récupérer la rampe selon la donnée
        ramp = None
        if isinstance(value, QgsColorRamp):
            ramp = value
        elif isinstance(value, str):
            try:
                # JSON sérialisé ?
                data = json.loads(value)
                ramp = self.deserialize(data)
            except Exception:
                # Peut-être un nom de rampe (ex: "viridis")
                ramp = self.ramp_by_name(value)
        elif isinstance(value, dict):
            ramp = self.deserialize(value)

        if not ramp:
            ramp = self.ramp_by_name(self.default_ramp)

        # Construire le gradient Qt
        grad = QLinearGradient(rect.left(), rect.top(), rect.right(), rect.top())

        # Ajouter color1 / stops / color2
        try:
            c1 = ramp.color1()
            grad.setColorAt(0.0, QColor(c1.red(), c1.green(), c1.blue()))

            # Ajouter les stops
            stops = ramp.stops() if hasattr(ramp, 'stops') else []
            for stop in stops:
                c = stop.color
                grad.setColorAt(stop.offset, QColor(c.red(), c.green(), c.blue()))

            c2 = ramp.color2()
            grad.setColorAt(1.0, QColor(c2.red(), c2.green(), c2.blue()))
        except Exception as e:
            grad.setColorAt(0.0, QColor("red"))
            grad.setColorAt(1.0, QColor("blue"))

        painter.fillRect(rect, QBrush(grad))

        # Bordure si sélectionnée
        if option.state & QStyle.State_Selected:
            pen = QPen(option.palette.highlight().color())
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(rect)
        painter.restore()

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)

    def ramp_by_name(self, name: str):
        """Récupérer une rampe par son nom depuis QgsStyle ou matplotlib"""
        style = QgsStyle.defaultStyle()
        ramp = style.colorRamp(name)
        if not name:
            return style.colorRamp("Viridis")

        try:
            ramp = style.colorRamp(name)
            if ramp:
                return ramp
        except Exception:
            pass
        try:
            return self.deserialize(name)
        except Exception:
            return style.colorRamp(self.default_ramp)

    def serialize(self, ramp: QgsColorRamp):
        """Sérialise un QgsColorRamp en dictionnaire"""
        if not ramp:
            return None
        return {
            "type": ramp.type(),
            "properties": ramp.properties()
        }

    def deserialize(self, data):
        """Désérialise un dictionnaire/string en QgsColorRamp"""
        if not data:
            return None

        # Si c'est déjà un ramp, le retourner tel quel
        if isinstance(data, QgsColorRamp):
            return data

        # Si c'est une string, tenter de parser le JSON ou le nom
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                # Pas du JSON, probablement un nom de rampe
                return self.ramp_by_name(data)

        # Maintenant c'est un dict
        ramp_type = data.get("type")
        props = data.get("properties", {})

        try:
            if ramp_type == "gradient":
                return QgsGradientColorRamp.create(props)
            elif ramp_type == "random":
                return QgsLimitedRandomColorRamp.create(props)
            elif ramp_type == "colorbrewer":
                return QgsColorBrewerColorRamp.create(props)
            elif ramp_type == "preset":
                return QgsPresetSchemeColorRamp.create(props)
            else:
                # Fallback générique
                return QgsGradientColorRamp.create(props)
        except Exception as e:
            return self.ramp_by_name(self.default_ramp)


class ComboBoxDelegate(QStyledItemDelegate):
    def __init__(self, choices, parent=None):
        super().__init__(parent)
        self.choices = choices  # liste de choix (ex: ["rouge", "vert", "bleu"])

    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        combo.addItems(self.choices)
        combo.setEditable(False)
        return combo

    def setEditorData(self, editor, index):
        value = str(index.data(Qt.EditRole) or "")
        i = editor.findText(value)
        if i >= 0:
            editor.setCurrentIndex(i)

    def setModelData(self, editor, model, index):
        value = editor.currentText()
        model.setData(index, value, Qt.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)


class PercentOrValueDelegate(QStyledItemDelegate):
    TYPE_ROLE = Qt.UserRole + 1

    def createEditor(self, parent, option, index):
        widget = QWidget(parent)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        line_edit = QLineEdit(widget)
        line_edit.setAlignment(Qt.AlignRight)
        line_edit.setObjectName("value_input")

        combo = QComboBox(widget)
        combo.addItems(["%", "valeur"])
        combo.setObjectName("type_selector")
        combo.setFixedWidth(70)

        layout.addWidget(line_edit)
        layout.addWidget(combo)

        return widget

    def setEditorData(self, editor, index):
        value = index.model().data(index, Qt.EditRole)
        value_type = index.model().data(index, self.TYPE_ROLE) or "value"

        line_edit = editor.findChild(QLineEdit, "value_input")
        combo = editor.findChild(QComboBox, "type_selector")

        # Set combobox
        combo.setCurrentIndex(0 if value_type == "percent" else 1)

        # Convertir la valeur en float
        try:
            f = float(value)
        except Exception as e:
            f = 0.0

        line_edit.setText(f"{f:.4f}")

    def setModelData(self, editor, model, index):
        line_edit = editor.findChild(QLineEdit, "value_input")
        combo = editor.findChild(QComboBox, "type_selector")

        # Lire la saisie
        try:
            strval = line_edit.text()
            if strval:
                strval = strval.replace("%", "")

            val = float(strval)
        except:
            val = 0.0

        # Stocker seulement le flottant
        value_type = "percent" if combo.currentIndex() == 0 else "value"
        model.setData(index, val, Qt.EditRole)
        model.setData(index, value_type, self.TYPE_ROLE)

    def paint(self, painter, option, index):
        value = index.model().data(index, Qt.DisplayRole)
        value_type = index.model().data(index, self.TYPE_ROLE) or "value"

        try:
            f = float(value)
            text = f"{f:.2f}% " if value_type == "percent" else f"{f:.2f}"
        except Exception as e:
            text = str(value)

        painter.save()
        painter.drawText(option.rect, Qt.AlignRight | Qt.AlignVCenter, text)
        painter.restore()


_dimensions = [
    'z',
    'intensity',
    'scanner_channel',
    'classification',
    'point_source_id',
    'gps_time',
    'red',
    'green',
    'blue'
]

_bins = {
    'z',
    'intensity',
    'scanner_channel',
    'classification',
    'point_source_id',
    'gps_time',
    'red',
    'green',
    'blue'
}

_counts = {
    "classification",
    "point_source_id"
}

_defaults_headers = {
    "filename",
    "bbox",
    "zrange",
    "directory"
}


class Calculator:

    def __init__(
            self,
            entry,
            delimiter=5_000_000,
            divider=5,
            resolution=0.1,
            bins=1000,
            _filter=None
    ):
        self.bound_or_chunk = delimiter
        self.divider = divider
        self.resolution = max(0.1, iface.mapCanvas().mapUnitsPerPixel())
        self.entry = entry
        self.bins = bins
        self.result = None
        self.low = 5  # Eliminer les 5% plus faibles valeurs
        self.high = 5  # Eliminer les 5% plus extrèmes valeurs
        self._filter = _filter

        # Nettoyer les filtres invalides
        if self._filter:
            self._filter = {
                k: v for k, v in self._filter.items()
                if k in _dimensions and isinstance(v, dict)
            }

    def _merge_histogram(self, hist1, bins1, hist2, bins2):
        """Fusionne deux histogrammes avec interpolation pour aligner les bins."""
        global_min = min(bins1[0], bins2[0])
        global_max = max(bins1[-1], bins2[-1])
        n_bins = max(len(hist1), len(hist2))
        common_bins = np.linspace(global_min, global_max, n_bins + 1)

        centers1 = (bins1[:-1] + bins1[1:]) / 2
        centers2 = (bins2[:-1] + bins2[1:]) / 2
        centers_common = (common_bins[:-1] + common_bins[1:]) / 2

        interp1 = np.interp(centers_common, centers1, hist1)
        interp2 = np.interp(centers_common, centers2, hist2)

        merged_hist = interp1 + interp2
        return merged_hist, common_bins

    def compute(self):
        values = dict()

        if not isinstance(self.bound_or_chunk, laspy.Bounds):
            with laspy.open(self.entry["filename"]) as las:
                for result in las.chunk_iterator(self.bound_or_chunk):
                    for dim in _dimensions:
                        # Enlever les valeurs extrêmes
                        raw_result = getattr(result, dim)

                        if dim in self._filter.keys():
                            # Gestion des différents types de filtres
                            rules = self._filter[dim]
                            local_mask = np.ones(len(raw_result), dtype=bool)  # Même longueur que le nombre de points
                            if "include" in rules and rules["include"] is not None:
                                local_mask &= np.isin(raw_result, rules["include"])
                            if "exclude" in rules and rules["exclude"] is not None:
                                local_mask &= ~np.isin(raw_result, rules["exclude"])
                            if "min" in rules and rules["min"] is not None:
                                local_mask &= raw_result >= rules["min"]
                            if "max" in rules and rules["max"] is not None:
                                local_mask &= raw_result <= rules["max"]
                            if "range" in rules and isinstance(rules["range"], (list, tuple)) and len(
                                    rules["range"]) == 2:
                                local_mask &= (raw_result >= rules["range"][0]) & (raw_result <= rules["range"][1])
                            pts = raw_result[local_mask]
                        else:
                            pts = raw_result

                        if not len(pts):
                            continue

                        # Récupérer le min max éliminer les extrêmes
                        min_, max_ = np.percentile(pts, [self.low, 100 - self.high])
                        if min_ > max_:
                            min_, max_ = max_, min_

                        # Liste des points à traiter
                        pts = pts[(pts >= min_) & (pts <= max_)]
                        _, tmp_ = np.histogram(pts, bins=self.bins + 1)
                        _tmp_hist = np.histogram(pts, bins=tmp_)
                        if dim not in values.keys():
                            values[dim] = {
                                "min": min_,
                                "max": max_,
                                "mediane": [float(np.median(pts))],
                                "bins": tmp_,
                                "histogram": _tmp_hist
                            }
                        else:
                            # Merger les statistiques
                            res_hist, res_bins = self._merge_histogram(
                                _tmp_hist, tmp_,
                                values[dim]["histogram"], values[dim]["bins"]
                            )

                            values[dim]["min"] = min(values[dim]["min"], min_)
                            values[dim]["max"] = max(values[dim]["max"], max_)
                            values[dim]["mediane"] = values[dim]["mediane"] + [float(np.median(pts))]
                            values[dim]["bins"] = res_bins
                            values[dim]["histogram"] = res_hist

            # Computer le mediane à la fin
            for key, value in values.items():
                values[key]["mediane"] = np.median(values[key]["mediane"])

            self.result = values
        else:
            # Lancer l'opération
            filename = self.entry["filename"]
            self.process_copc([filename, self.bound_or_chunk, self.resolution, self.bins, self._filter])

    def process_copc(self, args):
        """Traite un chunk - chaque process ouvre son propre reader COPC"""
        filename, chunk, resolution, bins, _filter = args
        try:
            with laspy.CopcReader.open(filename) as reader:
                # Récupérer le résultat
                result = reader.query(chunk, max(0.1, iface.mapCanvas().mapUnitsPerPixel()))
                if not result.x.shape[0]:
                    return 1

                values = dict()
                for dim in _dimensions:
                    try:
                        # Enlever les valeurs extrêmes
                        raw_result = getattr(result, dim)
                        if not len(raw_result):
                            continue

                        if self._filter and dim in self._filter.keys():
                            # Gestion des différents types de filtres
                            rules = self._filter[dim]
                            local_mask = np.ones(len(raw_result), dtype=bool)  # Même longueur que le nombre de points
                            if "include" in rules and rules["include"] is not None:
                                local_mask &= np.isin(raw_result, rules["include"])
                            if "exclude" in rules and rules["exclude"] is not None:
                                local_mask &= ~np.isin(raw_result, rules["exclude"])
                            if "min" in rules and rules["min"] is not None:
                                local_mask &= raw_result >= rules["min"]
                            if "max" in rules and rules["max"] is not None:
                                local_mask &= raw_result <= rules["max"]
                            if "range" in rules and isinstance(rules["range"], (list, tuple)) and len(
                                    rules["range"]) == 2:
                                local_mask &= (raw_result >= rules["range"][0]) & (raw_result <= rules["range"][1])

                            pts = raw_result[local_mask]
                        else:
                            pts = raw_result

                        if not len(pts):
                            continue

                        # Récupérer le min max éliminer les extrêmes
                        min_, max_ = np.percentile(pts, [self.low, 100 - self.high])
                        if min_ > max_:
                            min_, max_ = max_, min_

                        pts = pts[(pts >= min_) & (pts <= max_)]
                        _, bin_ = np.histogram(pts, bins=bins + 1)

                        values[dim] = {
                            "min": min_,
                            "max": max_,
                            "mediane": float(np.median(pts)),
                            "bins": bin_,
                            "histogram": np.histogram(pts, bins=bin_)
                        }
                    except Exception as e:
                        print(f"Erreur pour dimension {dim}: ", e)

                self.result = values
                return 0
        except Exception as e:
            print(f"Erreur globale: {e}")
            return 1


def process_chunk(arguments):
    """Traite un chunk - chaque process ouvre son propre reader COPC"""
    try:
        # Créer un calculatrice
        entry = arguments[0]
        delimiter = arguments[1]
        divider = arguments[2]
        bins = arguments[3]
        _filter = arguments[4]

        calc = Calculator(
            entry,
            delimiter=delimiter,
            divider=divider,
            bins=bins,
            _filter=_filter
        )

        # Computer la statistique
        calc.compute()
        return {entry["filename"]: calc.result} if calc.result else None
    except Exception as e:
        return None


class Statistic:
    def __init__(self, _filter: dict = None):
        self._entries = []

        self._chunk = 10_000_000
        self._divider = 1
        self._bound = None
        self._stat = None
        self._filter = _filter

    def __repr__(self):
        return self._stat.__str__()

    @property
    def entries(self):
        return self._entries

    @entries.setter
    def entries(self, values: list):
        # Ajouter les entrées
        for it in values:
            # Vérifier les types de données
            if not isinstance(it, dict):
                continue

            # Vérifier la validité de l'item
            if False in [h in it.keys() for h in _defaults_headers]:
                return

            # Rajouter les entrées
            self._entries += [it]

    def clear(self):
        self.entries = []
        self._stat = None

    def current(self):
        return self.bound, self._stat

    def compute(self):
        results = dict()
        if not self.entries:
            return

        items = [
            [entry, self.bound if self.bound else self.chunk, self.divider, 1500, self._filter] for entry in
            self.entries]

        with ThreadPoolExecutor(max_workers=8) as ex:
            vals = list(ex.map(process_chunk, items))
            for val in vals:
                if val:
                    results.update(val)

        self._stat = results.copy()
        return results

    @property
    def chunk(self):
        return self._chunk

    @chunk.setter
    def chunk(self, val: int):
        self._chunk = val

    @property
    def divider(self):
        return self._divider

    @divider.setter
    def divider(self, val: int):
        self._divider = val

    @property
    def bound(self):
        return self._bound

    @bound.setter
    def bound(self, rect: str):
        self._bound = None
        if isinstance(rect, str):
            self._bound: laspy.Bounds = self.bound_from_wkt(rect)
        elif isinstance(rect, laspy.Bounds):
            self._bound: laspy.Bounds = rect
        else:
            pass

    def merge(self):
        if not self._stat:
            return

        merged_results = {}
        for key, value in self._stat.items():
            for dim, val in value.items():
                hist, _ = val["histogram"]
                bins = val["bins"]

                if dim not in merged_results:
                    merged_results[dim] = {
                        "min": val["min"],
                        "max": val["max"],
                        "mediane": [val["mediane"]],
                        "bins": val["bins"],
                        "histogram": hist.copy(),
                        "files": Path(key).stem
                    }
                else:
                    merged_hist, merged_bins = self._merge_histogram(
                        merged_results[dim]["histogram"],
                        merged_results[dim]["bins"],
                        hist,
                        bins
                    )

                    merged_results[dim]["histogram"] = merged_hist
                    merged_results[dim]["bins"] = merged_bins
                    merged_results[dim]["min"] = min(val["min"], merged_results[dim]["min"])
                    merged_results[dim]["max"] = max(val["max"], merged_results[dim]["max"])
                    merged_results[dim]["mediane"].append(val["mediane"])
                    merged_results[dim]["files"] += ";" + Path(key).stem

        # Calculer la mediane de mediane des données
        for key, value in merged_results.items():
            merged_results[key]["mediane"] = np.median(merged_results[key]["mediane"])

        return merged_results

    def save(self, folder: str = None, merge: bool = False, extension: str = ".svg", export: bool = False):
        hist_files = []
        export_id = uuid.uuid4().hex
        if not self._stat or not folder or not os.path.exists(folder):
            return hist_files

        if merge:
            merged_results = self.merge()
        else:
            merged_results = {}

        # Créer un writer
        if export:
            layer: QgsVectorLayer = self.writer(os.path.join(folder, "histogram.shp"))
            layer.startEditing()
            try:
                geometry = QgsGeometry.fromRect(
                    QgsRectangle(
                        self.bound.mins[0],
                        self.bound.mins[1],
                        self.bound.maxs[0],
                        self.bound.maxs[1]
                    )
                )
            except Exception:
                export = False
                geometry = None
        else:
            layer = None

        def _save(_it: dict, _folder_name: str = ""):
            for k, v in _it.items():
                bns = v["bins"]
                try:
                    hst, _ = v["histogram"]
                except:
                    hst = v["histogram"]

                plt.figure(figsize=(8, 5))
                plt.hist(
                    x=bns[:-1],
                    bins=bns,
                    weights=hst,
                    histtype='bar',
                    edgecolor='black',
                    label=f"{k}"
                )

                # Afficher les medianes, min, max
                plt.axvline(v["min"], c="red", label=f"Minimum: {v['min']}")
                plt.axvline(v["max"], c="red", label=f"Maximum: {v['max']}")
                plt.axvline(v["mediane"], c="blue", label=f"Mediane: {v['mediane']}")

                plt.title(f"Histogramme '{k} [{export_id}]'")
                plt.xlabel(k)
                plt.ylabel("Nombre de points")
                plt.legend()

                # Enregistrer l'histogramme
                basename = f"histogram_{k}{extension}"

                # Créer le dossier au cas ou c'est inéxistant
                file_path = os.path.join(folder, export_id, _folder_name, basename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                # Sauvegarder l'histogramme
                plt.savefig(file_path)
                hist_files.append([file_path, v])
                plt.close()

                if layer:
                    feature = QgsFeature()
                    feature.setFields(layer.fields())
                    feature.setGeometry(geometry)
                    feature["id"] = uuid.uuid4().hex
                    feature["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    feature["dimension"] = k
                    feature["folder"] = export_id
                    feature["minimum"] = v["min"]
                    feature["maximum"] = v["max"]
                    feature["mediane"] = v["mediane"]

                    if _folder_name and _folder_name != "":
                        feature["files"] = _folder_name
                    else:
                        feature["files"] = v.get("files", "")

                    # Écrire la feature
                    layer.addFeature(feature)

        # Enregistrer les histogrammes
        data = merged_results if merge else self._stat
        for key, value in data.items():
            if merge:
                _save({key: value})
            else:
                for dim, val in value.items():
                    _save({dim: val}, _folder_name=Path(key).stem)

        if layer:
            layer.commitChanges()
            layer = None

        return hist_files

    def writer(self, output_path):
        try:
            # Récupérer le CRS du canvas
            crs = iface.mapCanvas().mapSettings().destinationCrs()
            # Créer les champs (attributs)
            fields = QgsFields()
            fields.append(QgsField("id", QVariant.String))
            fields.append(QgsField("date", QVariant.String))
            fields.append(QgsField("folder", QVariant.String))
            fields.append(QgsField("dimension", QVariant.String))
            fields.append(QgsField("minimum", QVariant.Double))
            fields.append(QgsField("maximum", QVariant.Double))
            fields.append(QgsField("mediane", QVariant.Double))
            fields.append(QgsField("files", QVariant.String, len=4000))

            layer = None
            if os.path.exists(output_path):
                layer = QgsVectorLayer(output_path, Path(output_path).stem, "ogr")

                # Vérifier le CRS
                if layer and layer.crs() != crs:
                    layer = None
                    QgsVectorFileWriter.deleteShapeFile(output_path)

                # Vérifier les champs
                if layer:
                    for field in fields:
                        if layer.fields().indexFromName(field.name()) == -1:
                            QgsVectorFileWriter.deleteShapeFile(output_path)
                            layer = None
                            break

            if layer is None:
                # Configurer l'écriture du shapefile
                writer = QgsVectorFileWriter(
                    output_path,
                    "UTF-8",
                    fields,
                    QgsWkbTypes.Polygon,
                    crs,
                    "ESRI Shapefile"
                )

                if writer.hasError() != QgsVectorFileWriter.NoError:
                    return None

                del writer
                layer = QgsVectorLayer(output_path, Path(output_path).stem, "ogr")
                layer.loadNamedStyle(os.path.join(UI_PATH, "histogram.qml"))
                layer.saveDefaultStyle()

            return layer

        except Exception as e:
            print(f"✗ Erreur: {str(e)}")
            return None

    def _merge_histogram(self, hist1, bins1, hist2, bins2):
        """Fusionne deux histogrammes avec interpolation pour aligner les bins."""
        global_min = min(bins1[0], bins2[0])
        global_max = max(bins1[-1], bins2[-1])
        n_bins = max(len(hist1), len(hist2))
        common_bins = np.linspace(global_min, global_max, n_bins + 1)

        centers1 = (bins1[:-1] + bins1[1:]) / 2
        centers2 = (bins2[:-1] + bins2[1:]) / 2
        centers_common = (common_bins[:-1] + common_bins[1:]) / 2

        interp1 = np.interp(centers_common, centers1, hist1)
        interp2 = np.interp(centers_common, centers2, hist2)

        merged_hist = interp1 + interp2
        return merged_hist, common_bins

    def bound_from_wkt(self, wkt: str):
        # Retirer "Polygon ((" et "))" puis séparer les points
        coords_text = wkt.replace("Polygon ((", "").replace("))", "")
        points = coords_text.split(",")

        # Convertir en floats
        coords = [tuple(map(float, p.strip().split())) for p in points]

        # Trouver min/max x et y
        xs, ys = zip(*coords)
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)

        # Créer laspy.Bounds
        return laspy.Bounds([minx, miny], [maxx, maxy])


class ZonalStatisticsWorker(QRunnable):

    def __init__(
            self, layer: str,
            save: bool = False,
            merge: bool = True,
            export_as_shp: bool = True,
            extension: str = ".png",
            folder=None,
            _filter: dict = None,
            box: QgsRectangle = None
    ):
        QRunnable.__init__(self)
        self.signals = WorkerSignals()
        self.save = save
        self.merge = merge
        self.export = export_as_shp
        self.extension = extension
        self.folder = folder
        self.result = None
        self._filters = _filter

        self.cloud = PointCloud(_filter=self._filters)
        self.cloud.file = layer

        # Récupérer l'extent du canvas
        box = box or iface.mapCanvas().extent()
        self.cloud.bound = QgsGeometry.fromRect(box).asWkt()

    def run(self):
        start = time.time()
        result = {
            "statistics": {},
            "histograms": [],
            "extent": ""
        }

        try:
            # Créer le parser
            self.signals.started.emit([], "lancement du calcul")
            results = self.cloud.compute(self.merge)
            duree = time.time() - start

            result["statistics"] = results
            self.signals.calulated.emit(results, f"{duree:.2f} Sec")
            if self.save:
                result["histograms"] = self.cloud.save_histogram(
                    export=self.export,
                    merge=self.merge,
                    extension=self.extension,
                    folder=self.folder
                )

        except Exception as e:
            self.signals.error.emit(f"Erreur lors du calcul: {e}")

        # Durée d'exécution de l'opération
        duree = time.time() - start
        self.result = result
        self.signals.finished.emit(result, f"{duree:.2f} Sec")


class PointCloud(QObject):
    computeFinished = pyqtSignal()
    histogramFinished = pyqtSignal()

    def __init__(self, _filter: dict = None):
        QObject.__init__(self, None)
        self._file = None
        self._data = []
        self._filter = _filter
        self._stat = Statistic(_filter=self._filter)

    @property
    def entries(self):
        return self._data

    def compute(self, merge: bool = False):
        result = self._stat.compute()
        self.computeFinished.emit()
        if merge:
            # Dimension comme clé
            return self._stat.merge()

        # Fichier comme clé
        return result

    def save_histogram(self, folder=None, merge: bool = False, extension: str = ".svg", export: bool = False):
        # Construire le dossier de sortie
        if not folder:
            folder = os.path.dirname(self.file) if self._file and os.path.exists(self.file) else None

        folder = os.path.join(folder, "Histogrammes")
        os.makedirs(folder, exist_ok=True)

        # Envoyer la liste des fichiers enreigistrés
        hist_files = self._stat.save(
            folder=folder,
            merge=merge,
            extension=extension,
            export=export
        )
        self.histogramFinished.emit()
        return hist_files

    @property
    def bound(self):
        return self._stat.bound

    @bound.setter
    def bound(self, box: str):
        # Calculer les statistiques
        self._stat.bound = box

    @property
    def file(self):
        return self._file if self._file and os.path.exists(self._file) else None

    @file.setter
    def file(self, f: str):
        # Parse file
        if (
                not os.path.exists(f)
                or Path(f).suffix not in [".vpc", ".laz", ".las"]
        ):
            self._data = []
            self._file = None
            self._stat.clear()
            return

        try:
            # Enregistrer le fichier
            self._file = f

            # Récupérer les paramètres
            if Path(self._file).suffix == ".vpc":
                self._data = []
                dire = os.path.dirname(self.file)

                # Lire le fichier
                with open(self._file, "r") as stream:
                    geojson = json.load(stream)

                # Récupérer les entités
                for feature in geojson.get("features", []):
                    asset = feature.get("assets", {}).get("data", {})
                    href = asset.get("href", "")
                    bbox = feature.get("properties", {}).get("proj:bbox", {})
                    if href.endswith(".copc.laz"):
                        file_path = (Path(dire) / Path(href)).resolve(strict=False)
                        self._data.append({
                            "filename": file_path.__str__(),
                            "bbox": laspy.Bounds(bbox[:3], bbox[3:]),
                            "zrange": [bbox[2], bbox[5]],
                            "directory": file_path.parent.__str__()
                        })

            # Traiter les fichiers autres que vpc
            elif Path(self._file).suffix in [".laz", ".las"]:
                self._data = []
                with laspy.open(self._file) as stream_:
                    header = stream_.header
                    self._data.append({
                        "filename": self._file,
                        "bbox": laspy.Bounds(header.mins, header.maxs),
                        "zrange": [header.z_min, header.z_max],
                        "directory": os.path.dirname(self._file)
                    })

            self._stat.entries = self.entries
        except Exception as e:
            print("ERREUR/", e)
            self._data = []
            self._file = None
            return []

    def intersects(self, b1: laspy.Bounds, b2: laspy.Bounds) -> bool:
        """
        Vérifie si deux bounding boxes se chevauchent sur tous les axes.
        """
        return bool(np.all((b1.mins <= b2.maxs) & (b1.maxs >= b2.mins)))

    def contains(self, outer: laspy.Bounds, inner: laspy.Bounds) -> bool:
        """
        Vérifie si la bounding box `outer` contient entièrement la bounding box `inner`.
        """
        return bool(np.all((outer.mins <= inner.mins) & (outer.maxs >= inner.maxs)))

