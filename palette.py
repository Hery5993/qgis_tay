import json
import os.path
import sys
import uuid

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QAbstractTableModel, Qt, QVariant, QModelIndex
from PyQt5.QtGui import QColor, QIcon
from PyQt5.QtWidgets import (
    QMainWindow, QStyledItemDelegate,
    QLineEdit, QSpinBox,
    QDoubleSpinBox, QComboBox,
    QHBoxLayout, QFileDialog, QMessageBox,
    QWidget, QPushButton, QColorDialog,
    QMenu, QAction, QApplication
)
from qgis._core import QgsApplication, QgsPointCloudClassifiedRenderer
from qgis.core import QgsPointCloudLayer
from qgis.utils import iface
from ._lib import CloudUtils


class Ui_Palette(object):
    def setupUi(self, Palette):
        Palette.setObjectName("Palette")
        Palette.resize(800, 391)
        self.centralwidget = QtWidgets.QWidget(Palette)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.tableView = QtWidgets.QTableView(self.centralwidget)
        self.tableView.setObjectName("tableView")
        self.gridLayout.addWidget(self.tableView, 0, 0, 1, 1)
        Palette.setCentralWidget(self.centralwidget)
        self.toolBar = QtWidgets.QToolBar(Palette)
        self.toolBar.setObjectName("toolBar")
        Palette.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionNew = QtWidgets.QAction(Palette)
        self.actionNew.setObjectName("actionNew")
        self.actionSave = QtWidgets.QAction(Palette)
        self.actionSave.setObjectName("actionSave")
        self.actionDelete = QtWidgets.QAction(Palette)
        self.actionDelete.setObjectName("actionDelete")
        self.actionLoad = QtWidgets.QAction(Palette)
        self.actionLoad.setObjectName("actionLoad")
        self.toolBar.addAction(self.actionLoad)
        self.toolBar.addAction(self.actionSave)
        self.toolBar.addAction(self.actionNew)
        self.toolBar.addAction(self.actionDelete)

        self.retranslateUi(Palette)
        QtCore.QMetaObject.connectSlotsByName(Palette)

    def retranslateUi(self, Palette):
        _translate = QtCore.QCoreApplication.translate
        Palette.setWindowTitle(_translate("Palette", "MainWindow"))
        self.toolBar.setWindowTitle(_translate("Palette", "toolBar"))
        self.actionNew.setText(_translate("Palette", "New Line"))
        self.actionNew.setToolTip(_translate("Palette", "New Line"))
        self.actionNew.setShortcut(_translate("Palette", "Ctrl+N"))
        self.actionSave.setText(_translate("Palette", "Save to file"))
        self.actionSave.setToolTip(_translate("Palette", "Save to file"))
        self.actionSave.setShortcut(_translate("Palette", "Ctrl+S"))
        self.actionDelete.setText(_translate("Palette", "Delete a line"))
        self.actionDelete.setToolTip(_translate("Palette", "Delete a line"))
        self.actionDelete.setShortcut(_translate("Palette", "Ctrl+D"))
        self.actionLoad.setText(_translate("Palette", "Load from file"))
        self.actionLoad.setToolTip(_translate("Palette", "Load from file"))
        self.actionLoad.setShortcut(_translate("Palette", "Ctrl+O"))


class Model(QAbstractTableModel):
    """
    Modèle de table dynamique basé sur une structure de headers
    """

    def __init__(self, palette_headers, dimension: str, rows=0, parent=None):
        super().__init__(parent)
        self.palette_headers = palette_headers
        self.column_names = list(palette_headers.keys())
        self.dimension = dimension
        self.current_file = None

        # Initialiser les données avec le nombre de lignes spécifié
        self._data = []
        for _ in range(rows):
            row_data = {}
            for col_name, col_config in palette_headers.items():
                # Utiliser la valeur par défaut
                if col_config.get("values"):
                    # Si c'est une liste, utiliser l'index par défaut
                    default_idx = col_config.get("default", 0)
                    row_data[col_name] = col_config["values"][default_idx]
                else:
                    # Sinon utiliser directement la valeur par défaut
                    row_data[col_name] = col_config.get("default")
            self._data.append(row_data)

    def rowCount(self, parent=None):
        return len(self._data)

    def columnCount(self, parent=None):
        return len(self.column_names)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return QVariant()

        row = index.row()
        col = index.column()

        if row >= len(self._data) or col >= len(self.column_names):
            return QVariant()

        col_name = self.column_names[col]
        col_config = self.palette_headers[col_name]
        value = self._data[row].get(col_name)
        datatype = col_config.get("datatype", None)

        # --- CheckStateRole pour les booléens (affichage de la checkbox)
        if role == Qt.CheckStateRole and datatype == "bool":
            return Qt.Checked if bool(value) else Qt.Unchecked

        # --- Affichage général
        if role == Qt.DisplayRole:
            # Pour les booléens, on peut afficher vide (la checkbox suffit),
            # ou "True"/"False" si tu préfères voir le texte.
            if datatype == "bool":
                return bool(value)

            # Vérifier s'il y a un suffixe personnalisé pour cette cellule
            suffix_key = f"{col_name}_suffix"
            suffix = self._data[row].get(suffix_key)
            if suffix is None:
                suffix = col_config.get("suffix", "")
            if suffix:
                return f"{value}{suffix}"
            return str(value) if value is not None else ""

        # --- EditRole renvoie la valeur brute (déjà OK)
        if role == Qt.EditRole:
            return value

        # --- Background pour color
        if role == Qt.BackgroundRole and col_config.get("datatype") == "color":
            try:
                return QColor(value)
            except:
                return QVariant()

        return QVariant()

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            if section < len(self.column_names):
                return self.column_names[section]
        return QVariant()

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemIsEnabled

        col = index.column()
        col_name = self.column_names[col]
        datatype = self.palette_headers[col_name].get("datatype", "")
        base = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if datatype == "bool":
            return base | Qt.ItemIsUserCheckable | Qt.ItemIsEditable

        # Pour les autres colonnes, editable si nécessaire
        return base | Qt.ItemIsEditable

    def setData(self, index, value, role=Qt.EditRole):
        if not index.isValid():
            return False

        row = index.row()
        col = index.column()

        if row >= len(self._data) or col >= len(self.column_names):
            return False

        col_name = self.column_names[col]
        col_config = self.palette_headers[col_name]
        datatype = col_config.get("datatype", "str")

        # --- Gérer la checkbox : role == Qt.CheckStateRole
        if datatype == "bool" and role == Qt.CheckStateRole:
            # value est Qt.Checked ou Qt.Unchecked
            new_val = (value == Qt.Checked)
            self._data[row][col_name] = new_val
            # notifier les rôles pertinents
            self.dataChanged.emit(index, index, [Qt.CheckStateRole, Qt.DisplayRole, Qt.EditRole])
            return True

        # --- Gérer l'EditRole classique (éditeurs)
        if role == Qt.EditRole:
            # Si la valeur est un tuple (valeur, suffixe), la gérer spécialement
            if isinstance(value, tuple) and len(value) == 2:
                actual_value, suffix = value
                try:
                    if datatype == "int":
                        self._data[row][col_name] = int(actual_value)
                    elif datatype == "float":
                        self._data[row][col_name] = float(actual_value)
                    else:
                        self._data[row][col_name] = actual_value
                except (ValueError, TypeError):
                    self._data[row][col_name] = actual_value
                suffix_key = f"{col_name}_suffix"
                self._data[row][suffix_key] = suffix
            else:
                try:
                    if datatype == "int":
                        self._data[row][col_name] = int(value)
                    elif datatype == "float":
                        self._data[row][col_name] = float(value)
                    elif datatype == "str":
                        self._data[row][col_name] = str(value)
                    elif datatype in ["file", "color"]:
                        self._data[row][col_name] = str(value)
                    elif datatype == "bool":
                        # dans le cas improbable où on envoie EditRole pour un bool
                        self._data[row][col_name] = bool(value)
                    else:
                        self._data[row][col_name] = value
                except (ValueError, TypeError):
                    self._data[row][col_name] = value

            self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
            return True

        return False

    def getData(self):
        """Retourne les données du modèle"""
        return self._data

    def getHeaders(self):
        """Retourne la configuration des headers"""
        return self.palette_headers

    def setAllData(self, data):
        """Remplace toutes les données du modèle"""
        self.beginResetModel()
        self._data = data
        self.endResetModel()

    def addRow(self):
        """Ajoute une nouvelle ligne avec valeurs par défaut"""
        row_position = len(self._data)
        # Utiliser QModelIndex() comme parent (pas QVariant)
        self.beginInsertRows(QModelIndex(), row_position, row_position)

        row_data = {}
        for col_name, col_config in self.palette_headers.items():
            if col_config.get("values"):
                default_idx = col_config.get("default", 0)
                row_data[col_name] = col_config["values"][default_idx]
            else:
                row_data[col_name] = col_config.get("default")

        self._data.append(row_data)
        self.endInsertRows()

    def removeRow(self, row):
        """Supprime une ligne"""
        if 0 <= row < len(self._data):
            from PyQt5.QtCore import QModelIndex
            self.beginRemoveRows(QModelIndex(), row, row)
            del self._data[row]
            self.endRemoveRows()
            return True
        return False

    def dump(self, file_path: str, dimension: str):
        if not file_path:
            return False

        try:
            export_data = {
                "headers": self.getHeaders(),
                "data": self.getData()
            }

            data = dict()
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r") as _stream:
                        _content = json.load(_stream)
                        if isinstance(_content, dict):
                            data = _content.copy()
                except:
                    pass

            with open(file_path, 'w', encoding='utf-8') as f:
                data[dimension] = export_data
                json.dump(data, f, ensure_ascii=False, indent=4)

            self.current_file = file_path
            self.dimension = dimension
            return True
        except Exception:
            return False

    def load(self, file_path: str, dimension: str):
        if not file_path or not os.path.exists(file_path):
            return False

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)

            # Vérifier que les données sont stockés sous forme d'une dictionaire
            if not isinstance(loaded_data[dimension], dict):
                return False

            # Valider la structure
            if dimension not in loaded_data:
                return False

            if "data" not in loaded_data[dimension]:
                raise ValueError("Le fichier doit contenir un dictionnaire avec 'data'")

            if "headers" not in loaded_data[dimension]:
                raise ValueError("Le fichier doit contenir un dictionnaire avec 'data'")

            # Charger les headers si présents
            self.palette_headers = loaded_data[dimension]["headers"].copy()
            self.column_names = list(loaded_data[dimension]["headers"].keys())

            # Charger les données
            self.dimension = dimension
            self.current_file = file_path
            self.setAllData(loaded_data[dimension]["data"])
            return True
        except Exception as e:
            return False

    def check_all(self, col):
        try:
            for row in range(len(self._data)):
                if isinstance(self._data[row][self.column_names[col]], bool):
                    self._data[row][self.column_names[col]] = True
            self.layoutChanged.emit()
        except Exception as e:
            print(self._data)
            return

    def uncheck_all(self, col):
        try:
            for row in range(len(self._data)):
                if isinstance(self._data[row][self.column_names[col]], bool):
                    self._data[row][self.column_names[col]] = False
            self.layoutChanged.emit()
        except Exception:
            return

    def invert_checked(self, col):
        try:
            for row in range(len(self._data)):
                value = self._data[row][self.column_names[col]]
                if isinstance(value, bool):
                    self._data[row][self.column_names[col]] = not value
            self.layoutChanged.emit()
        except Exception:
            return


class Delegate(QStyledItemDelegate):
    """
    Délégué intelligent qui s'adapte au type de données défini dans palette_headers
    """

    AVAILABLE_SUFFIXES = ["", "%"]

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.table_model = model

    def createEditor(self, parent, option, index):
        col = index.column()
        col_name = self.table_model.column_names[col]
        col_config = self.table_model.palette_headers[col_name]

        datatype = col_config.get("datatype", "str")
        default_val = col_config.get("default", None)
        values = col_config.get("values", [])
        suffix = col_config.get("suffix", None)
        precision = col_config.get("precision", 2)
        maximumf = col_config.get("maximum", sys.float_info.max)
        minimumf = col_config.get("minimum", -sys.float_info.max)
        minimumi = col_config.get("maximum", -9999999)
        maximumi = col_config.get("minimum", 9999999)
        suffixes = col_config.get("suffixes", self.AVAILABLE_SUFFIXES)
        file_filter = col_config.get("file_filter", "Tous les fichiers (*.*)")

        if datatype == "bool":
            return None

        # Si c'est une liste de valeurs, utiliser un QComboBox
        if values:
            editor = QComboBox(parent)
            editor.addItems([str(v) for v in values])
            editor.setFrame(False)
            return editor

        # Si c'est un fichier, créer un widget composite avec browse button
        if datatype == "file":
            container = QWidget(parent)
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(2)

            line_edit = QLineEdit(container)
            line_edit.setFrame(False)

            browse_btn = QPushButton("...", container)
            browse_btn.setMaximumWidth(30)
            browse_btn.clicked.connect(lambda: self._browseFile(line_edit, file_filter))

            layout.addWidget(line_edit, 1)
            layout.addWidget(browse_btn)
            container.line_edit = line_edit
            return container

        # Si c'est une couleur, créer un widget avec color picker
        if datatype == "color":
            container = QWidget(parent)
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(2)

            line_edit = QLineEdit(container)
            line_edit.setFrame(False)

            color_btn = QPushButton("🎨", container)
            color_btn.setMaximumWidth(30)
            color_btn.clicked.connect(lambda: self._pickColor(line_edit))

            layout.addWidget(line_edit, 1)
            layout.addWidget(color_btn)

            container.line_edit = line_edit
            return container

        # Pour les types numériques avec suffixe
        if suffix is not None:
            container = QWidget(parent)
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(2)

            # Créer l'éditeur approprié selon le type
            if datatype == "int":
                value_editor = QSpinBox(container)
                value_editor.setMinimum(minimumi)
                value_editor.setMaximum(maximumi)
            elif datatype == "float":
                value_editor = QDoubleSpinBox(container)
                value_editor.setMinimum(minimumf)
                value_editor.setMaximum(maximumf)
                value_editor.setDecimals(precision)
            else:
                value_editor = QLineEdit(container)

            # Créer le QComboBox pour le suffixe
            suffix_editor = QComboBox(container)
            suffix_editor.setEditable(True)
            suffix_editor.addItems(suffixes)
            suffix_editor.setCurrentText(suffix)
            suffix_editor.setMaximumWidth(80)

            layout.addWidget(value_editor, 3)
            layout.addWidget(suffix_editor, 1)

            container.value_editor = value_editor
            container.suffix_editor = suffix_editor
            return container

        # Types numériques sans suffixe
        if datatype == "int":
            editor = QSpinBox(parent)
            editor.setMinimum(minimumi)
            editor.setMaximum(maximumi)
            return editor

        if datatype == "float":
            editor = QDoubleSpinBox(parent)
            editor.setMinimum(minimumf)
            editor.setMaximum(maximumf)
            editor.setDecimals(precision)
            return editor

        # Par défaut, QLineEdit
        return QLineEdit(parent)

    def _browseFile(self, line_edit, file_filter: str):
        """Ouvre un dialogue pour sélectionner un fichier"""
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Sélectionner un fichier",
            line_edit.text(),
            file_filter
        )
        if file_path:
            line_edit.setText(file_path)

    def _pickColor(self, line_edit):
        """Ouvre un dialogue pour sélectionner une couleur"""
        current_color = QColor(line_edit.text()) if line_edit.text() else QColor(Qt.white)
        color = QColorDialog.getColor(current_color, None, "Choisir une couleur")
        if color.isValid():
            line_edit.setText(color.name())

    def setEditorData(self, editor, index):
        value = index.model().data(index, Qt.EditRole)
        row = index.row()
        col = index.column()
        col_name = self.table_model.column_names[col]
        col_config = self.table_model.palette_headers[col_name]

        # QComboBox simple
        if isinstance(editor, QComboBox):
            idx = editor.findText(str(value))
            if idx >= 0:
                editor.setCurrentIndex(idx)

        # Widget composite avec suffixe
        elif hasattr(editor, 'value_editor') and hasattr(editor, 'suffix_editor'):
            value_editor = editor.value_editor
            suffix_editor = editor.suffix_editor

            # Définir la valeur
            if isinstance(value_editor, (QSpinBox, QDoubleSpinBox)):
                value_editor.setValue(value if value is not None else 0)
            elif isinstance(value_editor, QLineEdit):
                value_editor.setText(str(value) if value is not None else "")

            # Définir le suffixe - vérifier d'abord s'il y a un suffixe personnalisé
            suffix_key = f"{col_name}_suffix"
            custom_suffix = self.table_model._data[row].get(suffix_key)

            if custom_suffix is not None:
                suffix = custom_suffix
            else:
                suffix = col_config.get("suffix", "")

            suffix_idx = suffix_editor.findText(suffix)
            if suffix_idx >= 0:
                suffix_editor.setCurrentIndex(suffix_idx)
            else:
                suffix_editor.setEditText(suffix)

        # Widget composite avec line_edit (file/color)
        elif hasattr(editor, 'line_edit'):
            editor.line_edit.setText(str(value) if value is not None else "")

        # SpinBox/DoubleSpinBox
        elif isinstance(editor, (QSpinBox, QDoubleSpinBox)):
            editor.setValue(value if value is not None else 0)

        # QLineEdit
        elif isinstance(editor, QLineEdit):
            editor.setText(str(value) if value is not None else "")

    def setModelData(self, editor, model, index):
        # QComboBox simple
        if isinstance(editor, QComboBox):
            value = editor.currentText()
            model.setData(index, value, Qt.EditRole)

        # Widget composite avec suffixe
        elif hasattr(editor, 'value_editor') and hasattr(editor, 'suffix_editor'):
            value_editor = editor.value_editor
            suffix_editor = editor.suffix_editor

            # Récupérer la valeur
            if isinstance(value_editor, (QSpinBox, QDoubleSpinBox)):
                value = value_editor.value()
            elif isinstance(value_editor, QLineEdit):
                value = value_editor.text()
            else:
                return

            # Récupérer le suffixe
            suffix = suffix_editor.currentText()

            # Sauvegarder les deux valeurs sous forme de tuple
            model.setData(index, (value, suffix), Qt.EditRole)

        # Widget composite avec line_edit (file/color)
        elif hasattr(editor, 'line_edit'):
            value = editor.line_edit.text()
            model.setData(index, value, Qt.EditRole)

        # SpinBox/DoubleSpinBox
        elif isinstance(editor, (QSpinBox, QDoubleSpinBox)):
            value = editor.value()
            model.setData(index, value, Qt.EditRole)

        # QLineEdit
        elif isinstance(editor, QLineEdit):
            value = editor.text()

            model.setData(index, value, Qt.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)


class Palette(QMainWindow, Ui_Palette):
    def __init__(self, dimension=uuid.uuid4().hex, title: str = uuid.uuid4().hex, parent=None):
        QMainWindow.__init__(self, parent=parent)
        self.setWindowTitle(title)
        self.setupUi(self)
        self.dimension = dimension
        self._headers = None
        self._layer = None
        self.model = None
        self.delegate = None

        self.actionNew.setIcon(QIcon(":/images/themes/default/symbologyAdd.svg"))
        self.actionSave.setIcon(QIcon(":/images/themes/default/mActionFileSaveAs.svg"))
        self.actionLoad.setIcon(QIcon(":/images/themes/default/mActionFileOpen.svg"))
        self.actionDelete.setIcon(QIcon(":/images/themes/default/mActionDeleteSelected.svg"))

        # Ajouter deux actions pour sauvegarder dans la couche
        self.toolBar.addSeparator()
        self.actionSaveLayer = QAction(
            QIcon(":/images/themes/default/mActionSaveAllEdits.svg"),
            "Enregistrer dans le fichier VPC",
            parent=self.toolBar
        )

        self.toolBar.addAction(self.actionSaveLayer)

        # Connecter les actions
        self.actionNew.triggered.connect(lambda: self.onNewLine())
        self.actionDelete.triggered.connect(lambda: self.onDeleteLine())
        self.actionSave.triggered.connect(lambda: self.onSaveToFile())
        self.actionLoad.triggered.connect(lambda: self.onLoadFromFile())
        self.actionSaveLayer.triggered.connect(lambda: self.save(file=self.layer, confirm=True))

        # Paramètrer le table view
        self.tableView.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tableView.customContextMenuRequested.connect(self._contextMenu)

        # Fichier courant
        self.current_file = None

    @property
    def layer(self):
        if self._layer:
            return self._layer.dataProvider().dataSourceUri()
        return None

    @layer.setter
    def layer(self, layer):
        try:
            layer_path = layer.dataProvider().dataSourceUri()
            if not isinstance(layer, QgsPointCloudLayer) or not layer_path.endswith(".vpc"):
                return

            self._layer = layer
        except Exception:
            return

    @property
    def headers(self):
        return self._headers

    @headers.setter
    def headers(self, value):
        if not self.dimension:
            self.dimension = uuid.uuid4().hex

        try:
            self._headers = value
            self.model = Model(value, self.dimension)
            self.tableView.setModel(self.model)

            # Appliquer le délégué intelligent
            self.delegate = Delegate(self.model)
            self.tableView.setItemDelegate(self.delegate)

            # # Ajuster la largeur des colonnes
            for i in range(self.model.columnCount()):
                width = self._headers[i].get("column_width", 150)
                self.tableView.setColumnWidth(i, width)

        except Exception:
            return

    def onNewLine(self):
        """Ajoute une nouvelle ligne au tableau"""
        self.model.addRow()
        new_row_index = self.model.rowCount() - 1
        if new_row_index >= 0:
            self.tableView.selectRow(new_row_index)
            # optionnel : assurer que la ligne apparaisse à l'écran
            self.tableView.scrollToBottom()

    def onDeleteLine(self):
        """Supprime la ligne sélectionnée"""
        selected_indexes = self.tableView.selectedIndexes()
        if not selected_indexes:
            QMessageBox.warning(
                self,
                "Aucune sélection",
                "Veuillez sélectionner une ligne à supprimer."
            )
            return

        # Demander confirmation
        reply = QMessageBox.question(
            self,
            "Confirmation",
            f"Voulez-vous vraiment supprimer {len(selected_indexes)} ligne(s) ?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Supprimer les lignes en commençant par la fin
            rows = sorted([index.row() for index in selected_indexes], reverse=True)
            for row in rows:
                self.model.removeRow(row)

    def onSaveToFile(self):
        """Sauvegarde les données dans un fichier JSON"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Sauvegarder le fichier",
            self.current_file or "",
            "Fichiers JSON (*.json);;Tous les fichiers (*)"
        )

        if not file_path:
            return

        try:
            self.model.dump(file_path, self.dimension)
            self.current_file = file_path
            QMessageBox.information(
                self,
                "Succès",
                f"Données sauvegardées dans:\n{file_path}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Erreur",
                f"Impossible de sauvegarder le fichier:\n{str(e)}"
            )

    def onLoadFromFile(self):
        """Charge les données depuis un fichier JSON"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Charger un fichier",
            self.current_file or "",
            "Fichiers JSON (*.json);;Tous les fichiers (*)"
        )

        if not file_path:
            return

        if self.model.load(file_path, self.dimension):
            QMessageBox.information(
                self,
                "Succès",
                f"Données chargées depuis:\n{file_path}"
            )
            self.current_file = file_path
        else:
            QMessageBox.critical(
                self,
                "Erreur",
                f"Impossible de charger le fichier"
            )

    @property
    def records(self):
        if not self.model:
            return []
        return self.model.getData()

    def _contextMenu(self, pos):
        if not self.model:
            return

        menu = QMenu(self)
        index = self.tableView.indexAt(pos)
        column = index.column()

        val = self.model.data(index)

        # Actions
        act_check_all = menu.addAction("Cocher tout")
        act_uncheck_all = menu.addAction("Décocher tout")
        act_invert = menu.addAction("Inverser")

        # Si pas sur une colonne ou si pas de checkbox -> désactiver
        if column < 0 or not isinstance(val, bool):
            act_check_all.setEnabled(False)
            act_uncheck_all.setEnabled(False)
            act_invert.setEnabled(False)
        else:
            act_check_all.setEnabled(True)
            act_uncheck_all.setEnabled(True)
            act_invert.setEnabled(True)

        # Exécution du menu
        action = menu.exec_(self.tableView.viewport().mapToGlobal(pos))
        if action == act_check_all:
            self.model.check_all(column)
        elif action == act_uncheck_all:
            self.model.uncheck_all(column)
        elif action == act_invert:
            self.model.invert_checked(column)

    @staticmethod
    def confirm(title, msg):
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

    def save(self, file, confirm=False):
        if confirm:
            if not Palette.confirm("Ecrire la palette", f"Voulez vous ecrire la palette dans le fichier: '{file}'?"):
                return

        if not file or not file.endswith(".vpc"):
            CloudUtils.gis_error("Sélectionner un fichier VPC Valide")
            return

        try:
            # Enregistrer la palette
            self.model.dump(file, self.dimension)
        except Exception as e:
            return


class ColorPalette(Palette):

    def __init__(self, dimension=uuid.uuid4().hex, title: str = uuid.uuid4().hex, parent=None):
        Palette.__init__(self, dimension, title, parent)

        self.actionLoadCurrentQML = QAction(
            QIcon(":images/themes/default/propertyicons/symbology.svg"),
            "Charger à partir du QML courant",
            parent=self.toolBar
        )

        self.toolBar.addAction(self.actionLoadCurrentQML)

        self.actionLoadQML = QAction(
            QIcon(":images/themes/default/propertyicons/stylepreset.svg"),
            "Charger à partir d'un QML",
            parent=self.toolBar
        )

        self.toolBar.addAction(self.actionLoadQML)
        self.actionLoadCurrentQML.triggered.connect(lambda: self._load_classified_qml())
        self.actionLoadQML.triggered.connect(lambda: self._load_qml())

    def _load_classified_qml(self, file: str = ""):
        try:
            renderer = None
            if file == "":
                layer = iface.activeLayer()
                if isinstance(layer, QgsPointCloudLayer):
                    renderer = layer.renderer()
            else:
                if isinstance(file, str) and file.endswith(".qml"):
                    layer = QgsPointCloudLayer("memory")
                    layer.loadNamedStyle(file)
                    if layer.renderer() and isinstance(layer.renderer(), QgsPointCloudClassifiedRenderer):
                        renderer = layer.renderer()

            if not renderer:
                CloudUtils.gis_error("Impossible de charger la symbologie")
                return

            if isinstance(renderer, QgsPointCloudClassifiedRenderer):
                renderer: QgsPointCloudClassifiedRenderer = renderer
                categories = renderer.categories()

                data = []
                for category in categories:
                    data.append(
                        {
                            self.dimension: category.value(),
                            "Nom": category.label(),
                            "Couleur": category.color().name(),
                            "Affichage Canvas": True,
                            "Affichage Profil": True,
                            "Plaquer": True
                        }
                    )

                self.model.setAllData(data)

        except Exception as e:
            CloudUtils.gis_error(f"Impossible de charger la palette depuis le QML courant: {e}")

    def _load_qml(self):
        file_path, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="Sélectionner un QML",
            filter="Qml (*.qml)"
        )

        if file_path:
            self._load_classified_qml(file_path)

#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = ColorPalette(dimension="PaletteStyle")
#     window.headers = {
#         "Dimensions": {
#             "values": ["Intensity", "Z", "NormalZ"],
#             "default": 1,
#             "suffix": None,
#             "prefix": None,
#             "datatype": "str"
#         },
#         "Méthode": {
#             "values": ["Min&Max", "Axe&Pas", "Cycle", "Symétrique", "Asymétrique"],
#             "default": 0,
#             "suffix": None,
#             "prefix": None,
#             "datatype": "str"
#         },
#         "Valeur 1": {
#             "values": [],
#             "default": 0,
#             "suffix": "%",
#             "prefix": None,
#             "datatype": "float"
#         },
#         "Valeur 2": {
#             "values": [],
#             "default": 0,
#             "suffix": "%",
#             "prefix": None,
#             "datatype": "float"
#         },
#         "Valeur 3": {
#             "values": [],
#             "default": 0,
#             "suffix": "%",
#             "prefix": None,
#             "datatype": "float"
#         },
#         "Valeur 4": {
#             "values": [],
#             "default": True,
#             "suffix": None,
#             "prefix": None,
#             "datatype": "bool"
#         },
#         "Style": {
#             "values": [],
#             "default": r"C:\Futurmap\Outils\test.qml",
#             "suffix": "",
#             "prefix": None,
#             "datatype": "file"
#         },
#     }
#
#     window.show()
#     sys.exit(app.exec_())
