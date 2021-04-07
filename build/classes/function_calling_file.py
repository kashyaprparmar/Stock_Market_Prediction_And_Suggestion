from NewPro import new_graph,predict_stock,print_three_comp,investNewSvm
import sys
print("investing is working")
#def onClickDownload():
#    print("dfunc working!!!")
#    new_graph.onClick3(ArgList[2],ArgList[3],ArgList[4],ArgList[5],ArgList[6])
#def onClickPredict():
#    predict_stock.pred_value(ArgList[2],ArgList[3])
ArgList=list(sys.argv)
if ArgList[1]=='DOWNLOAD':
    new_graph.onClick3(ArgList[2],ArgList[3],ArgList[4],ArgList[5],ArgList[6])
elif ArgList[1]=="PREDICT":
    predict_stock.pred_value(ArgList[2],ArgList[3])
elif ArgList[1]=="THREECOMP":
    print_three_comp.comp_three(ArgList[2],ArgList[3],ArgList[4],ArgList[5],ArgList[6],ArgList[7])
elif ArgList[1]=="INVEST":
    investNewSvm.svmCompNew()
