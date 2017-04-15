import org.json.JSONArray;
import org.json.JSONObject;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

/**
 * Created by Naga on 22-03-2017.
 */
@WebServlet(name = "TensorFlowImageDetails", urlPatterns = "/tensorFlowImageDetails")
public class TensorFlowImageDetails extends HttpServlet {

    @Override
    protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        StringBuilder buffer = new StringBuilder();
        BufferedReader reader = req.getReader();
        String line;
        while ((line = reader.readLine()) != null) {
            buffer.append(line);
        }
        String data = buffer.toString();
        System.out.println(data);
        String output = "";
        JSONObject params = new JSONObject(data);
        JSONObject result = params.getJSONObject("result");
        JSONObject parameters = result.getJSONObject("parameters");
		System.out.println(parameters);
		System.out.println(parameters);
        /* if (parameters.get("null").toString().equals("clear")){
			System.out.println(parameters);
			System.out.println(parameters);
            Data data_ob = Data.getInstance();
            JSONObject js1 = new JSONObject();
            JSONArray jsonArray = new JSONArray();
            jsonArray.put(" ");
            js1.put("data", jsonArray);
            data_ob.setData(js1.toString());
            data_ob.setFlag(true);
            JSONObject js = new JSONObject();
            js.put("speech", "screen is cleared");
            js.put("displayText", "screen is cleared");
            js.put("source", "image database");
            output = js.toString();
        }
        else{ */
			System.out.println(parameters);
			System.out.println(parameters);
			System.out.println(parameters);
            String classs = parameters.getString("flowers").toString();
            JSONObject js1 = new JSONObject();

            JSONArray jsonArray = new JSONArray();
            jsonArray.put("https://group.renault.com/wp-content/uploads/2014/04/renault-logan.jpg");
            jsonArray.put("http://www.evi-usa.com/Portals/0/products/md/vehicles_med_duty.jpg");
            jsonArray.put("https://cdn1.img.sputniknews.com/images/103854/62/1038546207.jpg");
			jsonArray.put("http://cloudlakes.com/data_images/gallery/pontiac-solstice/pontiac-solstice-04.jpg");
            jsonArray.put("http://www.wallcoo.net/nature/green_road/images/%5Bwallcoo.com%5D_road_and_tree_AP24138.jpg");
            jsonArray.put("http://cdn.homesandhues.com/images/gallery/11/0/11/15_STARPATH-Glow-in-the-Dark-Pathways_1-f.jpg");
            jsonArray.put("https://thumbs.dreamstime.com/x/parents-holds-hand-children-crossing-road-20570697.jpg");
            jsonArray.put("http://www.americantrails.org/photoGalleries/cool/30images/12.jpg");
            jsonArray.put("http://www.buildingtechnologies.siemens.com/bt/global/en/PublishingImages/total-building-solutions-key-visual-large.jpg");
            jsonArray.put("http://isoftdimension.com.sg/demo/ascendas/i10nstatic/mar_del_plata_cefira_building1.jpg");
            jsonArray.put("https://s-media-cache-ak0.pinimg.com/736x/51/eb/69/51eb6965fdb1ae474826ba0c8cf28edf.jpg");
			//            jsonArray.put("http://i64.tinypic.com/v2ur6w.jpg");
            js1.put("imageBase64", jsonArray);

           String url = "https://tensorflow-1717.herokuapp.com/";
			//String url = "http://ec2-34-200-227-114.compute-1.amazonaws.com/";
    //        String url = "https://mighty-headland-54930.herokuapp.com";
//            String url = "https://gentle-ocean-30903.herokuapp.com/api/predict";


//        String url = "https://image-details.herokuapp.com/imageDetails";
            URL obj = new URL(url);
            HttpURLConnection con = (HttpURLConnection) obj.openConnection();

            //add reuqest header
            con.setRequestMethod("POST");
            con.setRequestProperty("Content-Type", "application/json");
            con.setRequestProperty("Accept-Language", "en-US,en;q=0.5");
            String urlParameters = "sn=C02G8416DRJM&cn=&locale=&caller=&num=12345";

            // Send post request
            con.setDoOutput(true);
            DataOutputStream wr = new DataOutputStream(con.getOutputStream());
//        wr.writeBytes(urlParameters);
            wr.writeBytes(js1.toString());
            wr.flush();
            wr.close();

            int responseCode = con.getResponseCode();

            System.out.println("\nSending 'POST' request to URL : " + url);
            System.out.println("Post parameters : " + urlParameters);
            System.out.println("Response Code : " + responseCode);

            BufferedReader in = new BufferedReader(
                    new InputStreamReader(con.getInputStream()));
            String inputLine;
            StringBuffer response = new StringBuffer();

            while ((inputLine = in.readLine()) != null) {
                response.append(inputLine);
            }
            in.close();

            String[] c = response.toString().replace("[","").replace("]","").replace("'","").split(",");
            Data data_ob = Data.getInstance();
            JSONObject newData=new JSONObject();
            JSONArray js = new JSONArray();
            for (int i=0; i<c.length; i++){
                if (c[i].trim().equals(classs))
                    js.put(jsonArray.get(i));
            }
            newData.put("data", js);
            data_ob.setData(newData.toString());
            //print result
//            System.out.println(response.toString());
            JSONObject js2 = new JSONObject();
            js2.put("speech", "Images are displayed on the screen");
            js2.put("displayText", "Images are displayed on the screen");
            js2.put("source", "database");
            output = js2.toString();
        //}

        resp.setHeader("Content-type", "application/json");
        resp.getWriter().write(output);
    }
}
