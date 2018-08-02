import os
from sklearn.externals import joblib
from sklearn import tree

################################################################################################
################################################################################################
################################################################################################

class csharp_converter():
    
    def __init__(self,base_class):
        
        self.clf = base_class.clf
        self.dot_path = base_class.dot_path
        self.save_path = base_class.save_path
        self.clf_name = base_class.clf_name
        self.converted_name = base_class.converted_name

        self._make_dot_dat()
        self.all_pred_func = self._make_if_else()
        self._finishing_touch()

    def _finishing_touch(self):

        if self.clf_name == 'GradientBoostingRegressor':
            self.fw = open(self.save_path+self.converted_name+".cs","a+")
            self.fw.write("\n\tstatic void Main()\n\t{")
            self.fw.write("\n\t\tdouble ybar = "+str(self.clf.init_.mean)+";")
            self.fw.write("\n\t\tdouble lam = "+str(self.clf.get_params()['learning_rate'])+";")
            self.fw.write("\n\t\tdouble[] X = { };")
            self.fw.write("\n\t\tProgram program = new Program();")
            self.fw.write("\n\t\tdouble final_prediction = ybar + lam *("+self.all_pred_func[:-3]+");")
            self.fw.write("\n\t}")
            self.fw.write("\n}")
            self.fw.close()

        if self.clf_name == 'GradientBoostingClassifier':
            self.fw = open(self.save_path+self.converted_name+".cs","a+")
            self.fw.write("\n\tstatic void Main()\n\t{")
            self.fw.write("\n\t\tdouble prior = "+str(self.clf.init_.prior)+";")
            self.fw.write("\n\t\tdouble lam = "+str(self.clf.get_params()['learning_rate'])+";")
            self.fw.write("\n\t\tdouble[] X = { };")
            self.fw.write("\n\t\tProgram program = new Program();")
            self.fw.write("\n\t\tdouble prob = prior + lam *("+self.all_pred_func[:-3]+");")
            self.fw.write("\n\t\tprob = exp(prob) / (1 + exp(prob));")
            self.fw.write("\n\t}")
            self.fw.write("\n}")
            self.fw.close()

        if self.clf_name == 'RandomForestClassifier':
            self.fw = open(self.save_path+self.converted_name+".cs","a+")
            self.fw.write("\n\tstatic void Main()\n\t{")
            self.fw.write("\n\t\tdouble n_estimators = "+str(self.clf.get_params()['n_estimators'])+";")
            self.fw.write("\n\t\tdouble[] X = { };")
            self.fw.write("\n\t\tProgram program = new Program();")
            self.fw.write("\n\t\tdouble prob = ("+self.all_pred_func[:-3]+");")
            self.fw.write("\n\t\tprob = prob/n_estimators;")
            self.fw.write("\n\t}")
            self.fw.write("\n}")
            self.fw.close()

        if self.clf_name == 'RandomForestRegressor':
            self.fw = open(self.save_path+self.converted_name+".cs","a+")
            self.fw.write("\n\tstatic void Main()\n\t{")
            self.fw.write("\n\t\tdouble n_estimators = "+str(self.clf.get_params()['n_estimators'])+";")
            self.fw.write("\n\t\tdouble[] X = { };")
            self.fw.write("\n\t\tProgram program = new Program();")
            self.fw.write("\n\t\tdouble final_prediction = ("+self.all_pred_func[:-3]+");")
            self.fw.write("\n\t\tfinal_prediction = final_prediction/n_estimators;")
            self.fw.write("\n\t}")
            self.fw.write("\n}")
            self.fw.close()

    def _make_dot_dat(self):
        for file in os.listdir(self.dot_path):
            os.remove(self.dot_path+file)
        count = 0
        for t in self.clf.estimators_:
            try:
                t = t[0]
                count += 1
                name = self.converted_name+'_'+str(count).zfill(4)
                tree.export_graphviz(t,out_file = self.dot_path+name+'.dot')
                self.estimator_criterion = t.criterion
            except:
                count += 1
                name = self.converted_name+'_'+str(count).zfill(4)
                tree.export_graphviz(t,out_file = self.dot_path+name+'.dot')
                self.estimator_criterion = t.criterion
            
    def _make_if_else(self):

        os.chdir(self.dot_path)

        self.fw = open(self.save_path+self.converted_name+".cs","w")
        self.fw.write("class Program{\n")
        all_pred_func = ""

        for dot_file_name in os.listdir():
            f = open(dot_file_name,"r")
            parent = []
            child = []
            leaves = []
            node_descr = {}
            leaf_descr = {}
            parent_child = {}
            child_parent = {}
            for line in f.readlines():
                if "->" in line:
                    p = int(line.split("->")[0])
                    c = int(line.split("->")[1][:3])
                    parent.append(p)
                    child.append(c)
                    if p in parent_child:
                        parent_child[p].append(c)
                    else:
                        parent_child[p] = [c]
                    child_parent[c] = p
                if self.estimator_criterion == 'friedman_mse':
                    if 'label="X[' in line:
                        node_descr[int(line.split("[")[0])] = line.split('\\nfriedman_mse')[0].split('"')[-1]
                    if 'label="friedman_mse' in line:
                        leaf_descr[int(line.split("[")[0])] = float(line.split("value =")[-1].split('"')[0])
                elif self.estimator_criterion == 'mse':
                    if 'label="X[' in line:
                        node_descr[int(line.split("[")[0])] = line.split('\\nmse')[0].split('"')[-1]
                    if 'label="mse' in line:
                        leaf_descr[int(line.split("[")[0])] = float(line.split("value =")[-1].split('"')[0])
                elif self.estimator_criterion == 'gini':
                    if 'label="X[' in line:
                        node_descr[int(line.split("[")[0])] = line.split('\\ngini')[0].split('"')[-1]
                    if 'label="gini' in line:
                        leaf_descr[int(line.split("[")[0])] = float(line.split("value = [")[-1].split(']"')[0].split(',')[0]) - float(line.split("value = [")[-1].split(']"')[0].split(',')[1])
                    
            for node in child:
                if node not in parent:
                    leaves.append(node)

            self._make_beginning(dot_file_name)
            
            for leaf in leaves:
                self._make_text(leaf,leaf_descr,node_descr,parent_child,child_parent)
            self._make_end()
            all_pred_func +=  'program.' + dot_file_name.split('.')[0] + '(X) + '
            

        all_pred_func = all_pred_func[:-1]
        all_pred_func += ";\n"
        self.fw.close()
        return all_pred_func


    def _make_beginning(self,dot_file_name):
        tt = "\n\tdouble " + dot_file_name.split('.')[0] + "(double[] X) {\n\t\tdouble pred = 0;"
        self.fw.write(tt)

        
    def _make_text(self,leaf_node,leaf_descr,node_descr,parent_child,child_parent):
        value = leaf_descr[leaf_node]
        c = leaf_node
        flag = 1
        condition = "\n\t\tif ( "
        while flag == 1:
            try:
                p = child_parent[c]
                if parent_child[p].index(c) == 1:
                    condition += "(" + node_descr[p].split("<=")[0] + " > " + node_descr[p].split("<=")[1] + ") && "
                else:
                    condition += "(" + node_descr[p] + ") && "
                c = p
            except:
                flag = 0
        condition = condition[:-4] + "){ \n\t\t\t" + "pred = " + str(leaf_descr[leaf_node]) + ";\n\t\t}"
        self.fw.write(condition)

    def _make_end(self):
        tt = "\n\t\treturn pred;\n\t}\n"
        self.fw.write(tt)



class cpp_converter():
    
    def __init__(self,base_class):
        
        self.clf = base_class.clf
        self.dot_path = base_class.dot_path
        self.save_path = base_class.save_path
        self.clf_name = base_class.clf_name
        self.converted_name = base_class.converted_name

        self._make_dot_dat()
        self.all_pred_func = self._make_if_else()
        self._finishing_touch()

    def _finishing_touch(self):

        if self.clf_name == 'GradientBoostingRegressor':
            self.fw = open(self.save_path+self.converted_name+".cpp","a+")
            self.fw.write("int main()\n{")
            self.fw.write("\n\tdouble ybar = "+str(self.clf.init_.mean)+";")
            self.fw.write("\n\tdouble lam = "+str(self.clf.get_params()['learning_rate'])+";")
            self.fw.write("\n\tdouble X[] = { };")
            self.fw.write("\n\tdouble final_prediction = ybar + lam *("+self.all_pred_func[:-3]+");\n")
            self.fw.write("\n\treturn 0;\n}")
            self.fw.close()

        if self.clf_name == 'GradientBoostingClassifier':
            self.fw = open(self.save_path+self.converted_name+".cpp","a+")
            self.fw.write("int main()\n{")
            self.fw.write("\n\tdouble prior = "+str(self.clf.init_.prior)+";")
            self.fw.write("\n\tdouble lam = "+str(self.clf.get_params()['learning_rate'])+";")
            self.fw.write("\n\tdouble X[] = { };")
            self.fw.write("\n\tdouble prob = prior + lam *("+self.all_pred_func[:-3]+");")
            self.fw.write("\n\tprob = exp(prob) / (1 + exp(prob));")
            self.fw.write("\n\treturn 0;\n}")
            self.fw.close()

        if self.clf_name == 'RandomForestClassifier':
            self.fw = open(self.save_path+self.converted_name+".cpp","a+")
            self.fw.write("int main()\n{")
            self.fw.write("\n\tdouble n_estimators = "+str(self.clf.get_params()['n_estimators'])+";")
            self.fw.write("\n\tdouble X[] = { };")
            self.fw.write("\n\tdouble prob = ("+self.all_pred_func[:-3]+");")
            self.fw.write("\n\tprob = prob/n_estimators;")
            self.fw.write("\n\treturn 0;\n}")
            self.fw.close()

        if self.clf_name == 'RandomForestRegressor':
            self.fw = open(self.save_path+self.converted_name+".cpp","a+")
            self.fw.write("int main()\n{")
            self.fw.write("\n\tdouble n_estimators = "+str(self.clf.get_params()['n_estimators'])+";")
            self.fw.write("\n\tdouble X[] = { };")
            self.fw.write("\n\tdouble final_prediction = ("+self.all_pred_func[:-3]+");")
            self.fw.write("\n\final_prediction = final_prediction/n_estimators;")
            self.fw.write("\n\treturn 0;\n}")
            self.fw.close()

        


    def _make_dot_dat(self):
        for file in os.listdir(self.dot_path):
            os.remove(self.dot_path+file)
        count = 0
        for t in self.clf.estimators_:
            try:
                t = t[0]
                count += 1
                name = self.converted_name+'_'+str(count).zfill(4)
                tree.export_graphviz(t,out_file = self.dot_path+name+'.dot')
                self.estimator_criterion = t.criterion
            except:
                count += 1
                name = self.converted_name+'_'+str(count).zfill(4)
                tree.export_graphviz(t,out_file = self.dot_path+name+'.dot')
                self.estimator_criterion = t.criterion

    def _make_if_else(self):

        os.chdir(self.dot_path)

        self.fw = open(self.save_path+self.converted_name+".cpp","w")
        all_pred_func = ""

        for dot_file_name in os.listdir():
            f = open(dot_file_name,"r")
            parent = []
            child = []
            leaves = []
            node_descr = {}
            leaf_descr = {}
            parent_child = {}
            child_parent = {}
            for line in f.readlines():
                if "->" in line:
                    p = int(line.split("->")[0])
                    c = int(line.split("->")[1][:3])
                    parent.append(p)
                    child.append(c)
                    if p in parent_child:
                        parent_child[p].append(c)
                    else:
                        parent_child[p] = [c]
                    child_parent[c] = p
                if self.estimator_criterion == 'friedman_mse':
                    if 'label="X[' in line:
                        node_descr[int(line.split("[")[0])] = line.split('\\nfriedman_mse')[0].split('"')[-1]
                    if 'label="friedman_mse' in line:
                        leaf_descr[int(line.split("[")[0])] = float(line.split("value =")[-1].split('"')[0])
                elif self.estimator_criterion == 'mse':
                    if 'label="X[' in line:
                        node_descr[int(line.split("[")[0])] = line.split('\\nmse')[0].split('"')[-1]
                    if 'label="mse' in line:
                        leaf_descr[int(line.split("[")[0])] = float(line.split("value =")[-1].split('"')[0])
                elif self.estimator_criterion == 'gini':
                    if 'label="X[' in line:
                        node_descr[int(line.split("[")[0])] = line.split('\\ngini')[0].split('"')[-1]
                    if 'label="gini' in line:
                        leaf_descr[int(line.split("[")[0])] = float(line.split("value = [")[-1].split(']"')[0].split(',')[0]) - float(line.split("value = [")[-1].split(']"')[0].split(',')[1])

                    
            for node in child:
                if node not in parent:
                    leaves.append(node)

            self._make_beginning(dot_file_name)
            
            for leaf in leaves:
                self._make_text(leaf,leaf_descr,node_descr,parent_child,child_parent)
            self._make_end()
            all_pred_func +=  dot_file_name.split('.')[0] + '(X) + '
            

        all_pred_func = all_pred_func[:-1]
        all_pred_func += ";\n"
        self.fw.close()
        return all_pred_func


    def _make_beginning(self,dot_file_name):
        tt = "\ndouble " + dot_file_name.split('.')[0] + "(double X[]) {\n\tdouble pred;"
        self.fw.write(tt)

        
    def _make_text(self,leaf_node,leaf_descr,node_descr,parent_child,child_parent):
        value = leaf_descr[leaf_node]
        c = leaf_node
        flag = 1
        condition = "\n\tif ( "
        while flag == 1:
            try:
                p = child_parent[c]
                if parent_child[p].index(c) == 1:
                    condition += "(" + node_descr[p].split("<=")[0] + " > " + node_descr[p].split("<=")[1] + ") && "
                else:
                    condition += "(" + node_descr[p] + ") && "
                c = p
            except:
                flag = 0
        condition = condition[:-4] + "){ \n\t\t" + "pred = " + str(leaf_descr[leaf_node]) + ";\n\t}"
        self.fw.write(condition)

    def _make_end(self):
        tt = "\n\treturn pred;\n}\n"
        self.fw.write(tt)


class model_converter():

    def __init__(self,save_path = None,dot_path = None,object_path = None,
                 object_name = None, convert_language = 'C++',converted_name = None):

        if not save_path:
            self.save_path = os.getcwd() + '//save_folder//'
            try:
                os.mkdir(self.save_path)
            except:
                print("Folder already exists.")
        else:
            self.save_path = save_path
            try:
                os.mkdir(self.save_path)
            except:
                print("Folder already exists.")

        if not dot_path:
            self.dot_path = os.getcwd() + '//dot_folder//'
            try:
                os.mkdir(self.dot_path)
            except:
                print("Folder already exists.")
        else:
            self.dot_path = dot_path
            try:
                os.mkdir(self.dot_path)
            except:
                print("Folder already exists.")

        if not object_path:
            self.object_path = os.getcwd() + '//'
        else:
            self.object_path = object_path

        if not object_name:
            raise ValueError('Cannot move forward without name of the object to be converted.')
        else:
            self.object_name = object_name

        if not converted_name:
            self.converted_name = self.object_name.replace('.pkl','')
        else:
            self.converted_name = converted_name


        # if the model is already provided, we need not read it again
        self.clf = joblib.load(self.object_path + self.object_name)
        self.clf_name = self.clf.__class__.__name__
        self.convert_language = convert_language
        

    def convert_model(self):
        if self.convert_language == 'C#':
            cs = csharp_converter(self)
        elif self.convert_language == 'C++':
            cs = cpp_converter(self)
        self.converter_class =  cs

    def describe(self):
        attrs = vars(self)
        print("----------------------- All attributes of the class -----------------------")
        print('\n\n'.join("%s : %s" % item for item in attrs.items()))

            
################################################################################################
################################################################################################
################################################################################################


mc = model_converter(object_name = 'RF_clf.pkl',convert_language = 'C++')
mc.convert_model()

mc.describe()
