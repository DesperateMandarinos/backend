from flask import Flask, render_template,jsonify,request
import numpy as np
from joblib import load
import os
from flask_cors import CORS

#Cargar el modelo
dt=load("dt1.joblib")

#Generar el servidor (backend)
servidorWeb = Flask(__name__)

#Activar convenciones CORS para permitir la entrada de peticiones desde el frontend
CORS(servidorWeb, resources={r"/*": {"origins": "http://localhost:3000"}})

@servidorWeb.route("/holamundo", methods=['GET'])
def holamundo():
        return render_template('pagina1.html')

#Envio de datos a traves de JSON        
@servidorWeb.route('/modelo',methods=['POST'])        
def modeloPrediccion():
        #Procesar datos de entrada
        contenido=request.json
        print(contenido)
        datosEntrada = np.array([
                10, #contenido["OverallQual"],
                contenido["YearBuilt"],
                contenido["YearRemodAdd"],
                contenido["MasVnrArea"],
                5, #contenido["ExterQual"],
                5,#contenido["BsmtQual"],
                contenido["TotalBsmtSF"],
                5, #contenido["HeatingQC"],
                contenido["1stFlrSF"],
                contenido["GrLivArea"],
                contenido["FullBath"],
                5, #contenido["KitchenQual"],
                contenido["TotRmsAbvGrd"],
                contenido["Fireplaces"],
                5, #contenido["FireplaceQu"],
                contenido["GarageFinish"],
                contenido["GarageCars"],
                contenido["GarageArea"],
                #Identifies the general zoning classification of the sale.
                contenido["MSZoning_FV"],
                contenido["MSZoning_RH"],
                contenido["MSZoning_RL"],
                contenido["MSZoning_C"],
                contenido["MSZoning_RM"],
                #Physical locations within Ames city limits
                contenido["Neighborhood_Blmngtn"],
                contenido["Neighborhood_Blueste"],
                contenido["Neighborhood_BrDale"],
                contenido["Neighborhood_BrkSide"],
                contenido["Neighborhood_ClearCr"],
                contenido["Neighborhood_CollgCr"],
                contenido["Neighborhood_Crawfor"],
                contenido["Neighborhood_Edwards"],
                contenido["Neighborhood_Gilbert"],
                contenido["Neighborhood_IDOTRR"],
                contenido["Neighborhood_MeadowV"],
                contenido["Neighborhood_Mitchel"],
                contenido["Neighborhood_NAmes"],
                contenido["Neighborhood_NPkVill"],
                contenido["Neighborhood_NWAmes"],
                contenido["Neighborhood_NoRidge"],
                contenido["Neighborhood_NridgHt"],
                contenido["Neighborhood_OldTown"],
                contenido["Neighborhood_SWISU"],
                contenido["Neighborhood_Sawyer"],
                contenido["Neighborhood_SawyerW"],
                contenido["Neighborhood_Somerst"],
                contenido["Neighborhood_StoneBr"],
                contenido["Neighborhood_Timber"],
                contenido["Neighborhood_Veenker"],
                #Type of dwelling
                contenido["BldgType_1Fam"],
                contenido["BldgType_2fmCon"],
                contenido["BldgType_Duplex"],
                contenido["BldgType_Twnhs"],
                contenido["BldgType_TwnhsE"],
                #Style of dwelling
                contenido["HouseStyle_1.5Fin"],
                contenido["HouseStyle_1.5Unf"],
                contenido["HouseStyle_1Story"],
                contenido["HouseStyle_2.5Unf"],
                contenido["HouseStyle_2Story"],
                contenido["HouseStyle_SFoyer"],
                contenido["HouseStyle_SLvl"],
                #Exterior covering on house
                contenido["Exterior1st_AsbShng"],
                contenido["Exterior1st_AsphShn"],
                contenido["Exterior1st_BrkComm"],
                contenido["Exterior1st_BrkFace"],
                contenido["Exterior1st_CBlock"],
                contenido["Exterior1st_CemntBd"],
                contenido["Exterior1st_HdBoard"],
                contenido["Exterior1st_MetalSd"],
                contenido["Exterior1st_Plywood"],
                contenido["Exterior1st_Stucco"],
                contenido["Exterior1st_VinylSd"],
                contenido["Exterior1st_Wd Sdng"],
                contenido["Exterior1st_WdShing"],
                #Exterior covering on house (if more than one material)
                contenido["Exterior2nd_AsbShng"],
                contenido["Exterior2nd_AsphShn"],
                contenido["Exterior2nd_Brk Cmn"],
                contenido["Exterior2nd_BrkFace"],
                contenido["Exterior2nd_CBlock"],
                contenido["Exterior2nd_CmentBd"],
                contenido["Exterior2nd_HdBoard"],
                contenido["Exterior2nd_ImStucc"],
                contenido["Exterior2nd_MetalSd"],
                contenido["Exterior2nd_Plywood"],
                contenido["Exterior2nd_Stone"],
                contenido["Exterior2nd_Stucco"],
                contenido["Exterior2nd_VinylSd"],
                contenido["Exterior2nd_Wd Sdng"],
                contenido["Exterior2nd_Wd Shng"],
                #Masonry veneer type
                contenido["MasVnrType_BrkCmn"],
                contenido["MasVnrType_BrkFace"],
                contenido["MasVnrType_None"],
                contenido["MasVnrType_Stone"],
                #Type of foundation
                contenido["Foundation_BrkTil"],
                contenido["Foundation_CBlock"],
                contenido["Foundation_PConc"],
                contenido["Foundation_Slab"],
                contenido["Foundation_Stone"],
                contenido["Foundation_Wood"],
                #Garage location
                contenido["GarageType_2Types"],
                contenido["GarageType_Attchd"],
                contenido["GarageType_Basment"],
                contenido["GarageType_BuiltIn"],
                contenido["GarageType_CarPort"],
                contenido["GarageType_Detchd"],
                contenido["GarageType_NA_Valor"],
                #Type of sale
                contenido["SaleType_COD"],
                contenido["SaleType_CWD"],
                contenido["SaleType_Con"],
                contenido["SaleType_ConLD"],
                contenido["SaleType_ConLI"],
                contenido["SaleType_ConLw"],
                contenido["SaleType_New"],
                contenido["SaleType_Oth"],
                contenido["SaleType_WD"],
                #Condition of sale
                contenido["SaleCondition_Abnorml"],
                contenido["SaleCondition_AdjLand"],
                contenido["SaleCondition_Family"],
                contenido["SaleCondition_Alloca"],
                contenido["SaleCondition_Normal"],
                contenido["SaleCondition_Partial"]

        ])

        #utiizar el modelo
        resultado=dt.predict(datosEntrada.reshape(1,-1))

        return jsonify({'resultado':str(resultado[0])})

if __name__ == '__main__':
        servidorWeb.run(debug=False,host='0.0.0.0',port='8080')